import matplotlib.pyplot as plt
import numpy as np
from huggingface_hub import hf_hub_download
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import torch
from transformers import CLIPProcessor, CLIPModel
import torch.nn.functional as F
import networkx as nx
from torch_geometric.data import Data
from PIL import Image
import matplotlib.patches as patches
from matplotlib.lines import Line2D


class Generator:
    def __init__(self, model_name='sam', model_type='vit_b', repo_id="xingren23/comfyflow-models",
                 filename="sams/sam_vit_b_01ec64.pth", chkpt_path=None):
        """
        Initializes the Generator class.

        Args:
            model_name (str): The name of the model to be used. Defaults to sam
            model_type (str): The variant of the SAM model. Defaults to 'vit_b
            repo_id (str): The hugging face repository ID for downloading the model checkpoint.
            filename (str): The filename of the model checkpoint.
            chkpt_path (str, optional): The path to the checkpoint file. Defaults to None.
        """
        self.model_name = model_name
        self.repo_id = repo_id
        self.filename = filename
        self.chkpt_path = chkpt_path
        self.model_type = model_type
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def download_checkpoints(self):
        """
        Downloads the model checkpoints from the repository.

        Returns:
            chkpt_path (str): The path to the downloaded checkpoint file.
        """
        if self.chkpt_path is None:
            if self.model_name == 'sam':
                try:
                    self.chkpt_path = hf_hub_download(repo_id=self.repo_id, 
                                                      filename=self.filename)
                except Exception as e:
                    print("An error occurred while downloading the checkpoint:", e)
                    
        # Return the checkpoint path whether it was just downloaded or already set
        return self.chkpt_path
    
    def SAM(self):
        """
        Loads the SAM model with the checkpoint.

        Returns:
            SamAutomaticMaskGenerator: The SAM mask generator.
        """
        chkpt_path = self.download_checkpoints()
        sam = sam_model_registry[self.model_type](checkpoint=chkpt_path)
        sam.to(self.device)

        return SamAutomaticMaskGenerator(sam)
    
    def generate_masks(self, mask_generator, image):
        """
        Generates masks for the given image using the SAM mask generator.

        Args:
            mask_generator (SamAutomaticMaskGenerator): The SAM mask generator.
            image (numpy.ndarray): The image for which to generate masks.

        Returns:
            dict: A dictionary containing the generated segments.
        """
        self.image = image
        segment_dict = mask_generator.generate(image)
        return segment_dict
    
    def plot_segments(self, segment_dict, figsize=(16, 16), alpha=0.5, bbox=False):
        """
        Plots the segments on the image.

        Args:
            segment_dict (dict): The dictionary of generated segments.
            figsize (tuple, optional): The size of the figure. Defaults to (16, 16).
            alpha (float, optional): The transparency of the segment masks. Defaults to 0.5.
            bbox (bool, optional): Whether to plot bounding boxes around segments. Defaults to False.
        """
        fig,axes = plt.subplots(1,2, figsize=figsize)
        axes[0].imshow(self.image)
        ax = axes[1]

        
        sorted_result = sorted(segment_dict, key=(lambda x: x['area']), reverse=True)
        # Plot for each segment area
        for val in sorted_result:
            mask = val['segmentation']
            img = np.ones((mask.shape[0], mask.shape[1], 3))
            color_mask = np.random.random((1, 3)).tolist()[0]
            for i in range(3):
                img[:,:,i] = color_mask[i]
            ax.imshow(np.dstack((img, mask*alpha)))
        
        if bbox:
            for val in sorted_result:
                rect = patches.Rectangle((val["bbox"][0], val["bbox"][1]), val["bbox"][2], val["bbox"][3], 
                                        linewidth=1, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
        plt.show()
      
class SegmentGraph:
    def __init__(self, segment_dict, image_rgb, object_return=None, plot_topo = True, semantic=True, model_name="openai/clip-vit-base-patch32"):
        """
        Initialize the SegmentGraph class.

        Args:
            segment_dict (dict): Dictionary containing segmentation data.
            image_rgb (np.ndarray): RGB image array.
            object_return (str): The format of the returned object ('PyG' for PyTorch Geometric).
            plot_topo (bool): Flag to plot topology.
            semantic (bool): Flag to use semantic information.
            model_name (str): Name of the CLIP model to be used.
        """
        self.segment_dict = segment_dict
        self.plot_topo = plot_topo
        self.semantic=semantic
        self.object_return = object_return
        self.image_rgb = image_rgb
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if semantic:
            self.clip_model = CLIPModel.from_pretrained(model_name).to(self.device)
            self.clip_processor = CLIPProcessor.from_pretrained(model_name)

    def get_topology(self, dist_thres, area_percent=0.5, add_to_bbox=50):
        """
        Get the topology of the image based on segments and distance threshold.

        Args:
            dist_thres (float): Distance threshold for connecting nodes in the graph.
            area_percent (float): Percentage of area to be ignored.
            add_to_bbox (int): Size of bounding box 

        Returns:
            tuple: Graph object and centroids of the segments.
        """
        
        h, l = self.image_rgb.shape[0], self.image_rgb.shape[1]
        ign_area = (h*l)*area_percent/100


        self.filter_seg = []
        for res in self.segment_dict:
            if res["area"] < ign_area:
                continue
            else:
                self.filter_seg.append(res)


        filter_images = []
        for img in self.filter_seg:
            bbox = [int(x) for x in img['bbox']]
            crop_img = self.image_rgb[max(bbox[1] - add_to_bbox, 0):min(bbox[1] + bbox[3] + add_to_bbox, h),
                                max(bbox[0] - add_to_bbox, 0):min(bbox[0] + bbox[2] + add_to_bbox, l)]
            filter_images.append(crop_img)
        

        segment_images = []
        for img in filter_images:
            segment_images.append(Image.fromarray(np.uint8(img)))


        centroids = []
        for seg in self.filter_seg:
            bbox = seg["bbox"]
            x = bbox[0] + bbox[2]/2
            y = bbox[1] + bbox[3]/2
            centroids.append([x, y])
        
        nodes = []
        if self.semantic:
            inputs = self.clip_processor(images=segment_images, return_tensors="pt", padding=True)
            inputs = {key: value.to(self.clip_model.device) for key, value in inputs.items()}
            with torch.no_grad():
                self.image_embeddings = self.clip_model.get_image_features(**inputs).cpu()

            for cent, embed in zip(centroids, self.image_embeddings):
                node = {}
                node["centroid"] = cent
                node["embedding"] = embed
                nodes.append(node)
        else:
            nodes = []
            for cent in centroids:
                node = {}
                node["centroid"] = cent
                nodes.append(node)
        
        G = self.create_topology_graph(nodes)

        
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                dist = np.linalg.norm(np.array(nodes[i]['centroid']) - np.array(nodes[j]['centroid']))
                
                if dist < dist_thres:
                    G.add_edge(i, j, weight=dist)
                    
        self.plot_topology(G, centroids)

        if self.object_return == 'PyG':
            # Convert to PyTorch Geometric Data object
            edge_index = torch.tensor(list(G.edges), dtype=torch.long).t().contiguous()
            x = torch.stack([segment['embedding'] for segment in nodes])
            data = Data(x=x, edge_index=edge_index)
            return data, centroids
        
        return G, centroids
        

    def create_topology_graph(self, nodes):
        """
        Args:
            nodes with centroids and embeddings.

        Returns:
            networkx.Graph: Graph with nodes and edges.
        """
        # Create graph
        G = nx.Graph()

        # Add nodes
        for idx, segment in enumerate(nodes):
            G.add_node(idx, centroid=segment['centroid'], embedding=segment['embedding'])
        
        return G
    
    def plot_topology(self, graph, centroids, node_size=50, node_color='blue', alpha=0.8, bbox=False, seg=True):
        """
        Plot the topology graph on the image.

        Args:
            graph (networkx.Graph): Graph object to plot.
            centroids (list): List of centroids of the segments.
            node_size (int): Size of the nodes.
            node_color (str): Color of the nodes.
            alpha (float): Transparency level of nodes and edges.
            bbox (bool): Flag to plot bounding boxes.
            seg (bool): Flag to plot segment areas.
        """
        # Begin the plot
        fig, ax = plt.subplots()
        ax.imshow(self.image_rgb)
        
        if seg:
            sorted_result = sorted(self.filter_seg, key=(lambda x: x['area']), reverse=True)
            # Plot for each segment area
            for val in sorted_result:
                mask = val['segmentation']
                img = np.ones((mask.shape[0], mask.shape[1], 3))
                color_mask = np.random.random((1, 3)).tolist()[0]
                for i in range(3):
                    img[:,:,i] = color_mask[i]
                ax.imshow(np.dstack((img, mask*0.5)))
        if bbox:
            sorted_result = sorted(self.filter_seg, key=(lambda x: x['area']), reverse=True)
            for val in sorted_result:
                rect = patches.Rectangle((val["bbox"][0], 
                                        val["bbox"][1]),  
                                        val["bbox"][2], 
                                        val["bbox"][3], 
                                        linewidth=1,
                                        edgecolor='r',
                                        facecolor='none'
                                    )
                ax.add_patch(rect)

        positions = {i : (x, y) for i, (x, y) in enumerate(centroids)}
        # Draw the nodes (you can skip this if you only want to visualize the edges)
        nx.draw_networkx_nodes(graph, pos=positions, node_size=node_size, node_color=node_color, alpha=alpha)

        # Now draw the edges
        if len(graph.nodes) > 1: 
            nx.draw_networkx_edges(graph, pos=positions, edge_color='green', alpha=0.5)

        plt.axis('off')  
        plt.show()  
                    
    def get_similar_embeds(self, query_entity, img_embeds):
        """
        Args:
            query_entity (str): The query text.
            img_embeds (torch.Tensor): Image embeddings.

        Returns:
            imilarity scores and dot product of embeddings.
        """
        inputs = self.clip_processor(text=query_entity, return_tensors="pt", padding=True)
        inputs = {key: value.to(self.clip_model.device) for key, value in inputs.items()}
        with torch.no_grad():
            text_embeddings = self.clip_model.get_text_features(**inputs).cpu()
            
        similarities = F.cosine_similarity(img_embeds, text_embeddings, dim=1)
        
        return similarities, img_embeds@text_embeddings.T
    

    def query_segment_graph(self, graph, query_pair, node_idx=None, k_nearest=None, beta=None, show_legend=False):
        """
        Args:
            graph (networkx.Graph): The segment graph.
            query_pair (tuple): Pair of environment and target query text.
            node_idx (int): Specific node index to highlight.
            k_nearest (int): Number of nearest nodes to highlight.
            beta (int): Number of top similar nodes to consider.
            show_legend (bool): Flag to show legend on the plot.
        """
        env, target = query_pair

        # Env Embeddings
        similarities, _ = self.get_similar_embeds(env, self.image_embeddings)
        similarities, env_index = torch.topk(similarities, k=1, largest=True)

        # Dijkstra
        path_lengths = nx.single_source_dijkstra_path_length(graph, int(env_index))

        if len(graph.nodes) > 1:
            node_dist_pairs = list(path_lengths.items())[1:]
        else:
            node_dist_pairs = list(path_lengths.items())

        close_nodes = []
        for i, _ in node_dist_pairs:
            
            bbox = self.filter_seg[i]["bbox"]
            x = bbox[0] + bbox[2]/2
            y = bbox[1] + bbox[3]/2
            close_nodes.append((x, y))

        # Target Embeddings
        img_embeddings = [self.image_embeddings[i] for i, _ in node_dist_pairs]
        img_embeds = torch.stack(img_embeddings)
        
        similarities, sim_tensor = self.get_similar_embeds(target, img_embeds=img_embeds)

        if beta is not None:
            # Get the indices of the top_k highest similarities
            similarities, indices = torch.topk(similarities, k=len(img_embeds), largest=True)
            index = node_dist_pairs[min(indices[:beta])][0]
        else:
            try:
                # Get the indices with similarity higher than mean and std
                sim_tensor = sim_tensor.squeeze(1).tolist()
                indices = self.find_large_values_squared(data=sim_tensor)
                index = node_dist_pairs[min(indices)][0]
            except:
                similarities, indices = torch.topk(similarities, k=len(img_embeds), largest=True)
                beta = 2 if len(node_dist_pairs) >= 2 else 1
                index = node_dist_pairs[min(indices[:beta])][0]
        
        fig, ax = plt.subplots()
        ax.imshow(self.image_rgb)
        
        if show_legend:
            # Create a legend
            legend_handles = [
                Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Env'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='purple', markersize=10, label='Target') 
            ]
            
            plt.legend(handles=legend_handles, loc='upper right', bbox_to_anchor=(1.05, 1.05), borderaxespad=0.)

        bbox = self.filter_seg[int(index)]["bbox"]
        x = bbox[0] + bbox[2]/2
        y = bbox[1] + bbox[3]/2

        bbox = self.filter_seg[int(env_index)]["bbox"]
        x1 = bbox[0] + bbox[2]/2
        y1 = bbox[1] + bbox[3]/2
        

        plt.plot(x, y, marker='o', color='purple')
        plt.plot(x1, y1, 'ro')


        if node_idx is not None:
            bbox = self.filter_seg[int(node_idx)]["bbox"]
            x2 = bbox[0] + bbox[2]/2
            y2 = bbox[1] + bbox[3]/2
            plt.plot(x2, y2, 'go')
            

        if k_nearest is not None:
            if k_nearest > len(close_nodes):
                k_nearest = len(close_nodes)
            close_nodes = close_nodes[:k_nearest]
            for x, y in close_nodes:
                plt.plot(x-10, y-10, 'yo')

        result = {
            "env_node_idx" : env_index,
            "target_node_idx" : index,
            "closest_nodes" : indices,
            "shortest_paths" : node_dist_pairs,
            "embeddings_dist" : [node_dist_pairs[idx][0] for idx in indices],
        }

        return result
    
    def find_large_values_squared(self, data):
        squared_data = [x**2 for x in data]
        mean = np.mean(squared_data)
        std_dev = np.std(squared_data)
        large_values = [i for i, x in enumerate(squared_data)if x > mean + 1.3*std_dev]
        return large_values