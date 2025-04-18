# utils/gallery.py
import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict # Keep if needed for other gallery types

# Import re-ranking functionality from the sibling module
from .reranking import perform_re_ranking, is_reranking_available

def get_unique_id(existing_ids_set, next_id_counter):
    """Finds the next available unique ID starting from the counter."""
    while next_id_counter in existing_ids_set:
        next_id_counter += 1
    return next_id_counter

class ReIDGallery:
    def __init__(self, config):
        """
        Initializes the ReID Gallery using optimized tensor storage.

        Args:
            config (dict): The 'reid' section of the application configuration.
        """
        self.config = config
        self.device = config['device']
        self.similarity_threshold = config['similarity_threshold']
        self.use_re_ranking = config['use_re_ranking']
        self.k1 = config['k1']
        self.k2 = config['k2']
        self.lambda_value = config['lambda_value']
        self.feature_dim = config['expected_feature_dim']

        # --- Initialize Gallery Storage (Tensor-based) ---
        # Stores features directly on the target device
        self.gallery_features_tensor = torch.empty((0, self.feature_dim), dtype=torch.float32, device=self.device)
        # Python list to store corresponding IDs in the same order as features in the tensor
        self.db_ids_list = []
        # Counter for assigning new IDs
        self.next_person_id_counter = 0

        print(f"ReID Gallery (Tensor Optimized) initialized on device '{self.device}'.")
        print(f"Similarity Threshold: {self.similarity_threshold}, Use Re-ranking: {self.use_re_ranking}")
        if self.use_re_ranking and not is_reranking_available():
            print("Warning: Re-ranking is enabled in config but the function is unavailable. Disabling.")
            self.use_re_ranking = False # Override if function not loaded

    def _update_gallery(self, new_feature_tensor, new_id):
        """Appends a new feature tensor (N, D) and corresponding ID(s) to the gallery."""
        if new_feature_tensor is None or new_feature_tensor.shape[0] == 0:
             return False # Nothing to add

        # Ensure feature tensor is 2D (N, D)
        if new_feature_tensor.dim() == 1:
            new_feature_tensor = new_feature_tensor.unsqueeze(0)
        elif new_feature_tensor.dim() != 2:
            print(f"Error: Cannot add feature tensor with dimension {new_feature_tensor.dim()}")
            return False

         # Ensure correct device and dtype
        new_feature_tensor = new_feature_tensor.to(self.device, dtype=self.gallery_features_tensor.dtype)

        # Concatenate the new feature tensor(s)
        self.gallery_features_tensor = torch.cat(
            (self.gallery_features_tensor, new_feature_tensor), dim=0
        )

        # Append the new ID(s) to the list
        if isinstance(new_id, (list, tuple)):
             self.db_ids_list.extend(new_id)
        else:
             self.db_ids_list.append(new_id)

        # print(f"Gallery updated. Size: {self.gallery_features_tensor.shape[0]} features.") # Optional log
        return True

    @torch.no_grad()
    def compare_features(self, query_features_batch):
        """
        Compares a batch of query features against the current gallery.

        Args:
            query_features_batch (torch.Tensor): Batch of N query features (N, D).

        Returns:
            tuple: (best_match_indices, best_match_scores, initial_similarities)
                   best_match_indices (Tensor N): Index in the gallery of the best match for each query.
                   best_match_scores (Tensor N): Score of the best match (similarity or distance).
                   initial_similarities (Tensor N, M): Cosine similarities between query and gallery.
                   Returns (None, None, None) if gallery is empty or input is invalid.
        """
        if query_features_batch is None or query_features_batch.shape[0] == 0:
            return None, None, None

        current_gallery_size = self.gallery_features_tensor.shape[0]
        if current_gallery_size == 0:
            return None, None, None # Cannot compare with empty gallery

        # Ensure query features are on the correct device and dtype
        query_features_batch = query_features_batch.to(self.device, dtype=self.gallery_features_tensor.dtype)

        # --- Calculate initial cosine similarities ---
        # query (N, D), gallery (M, D) -> similarity (N, M)
        # Use broadcasting for efficiency
        initial_similarities = F.cosine_similarity(
            query_features_batch.unsqueeze(1), # Shape: (N, 1, D)
            self.gallery_features_tensor.unsqueeze(0), # Shape: (1, M, D)
            dim=2 # Compare across feature dimension D
        ) # Result shape (N, M)

        # --- Apply Re-ranking if enabled ---
        final_scores_or_distances = initial_similarities
        is_distance = False # Flag: True if scores represent distance (lower is better)

        if self.use_re_ranking:
            # perform_re_ranking returns distances (lower is better) or None
            dist_mat = perform_re_ranking(
                query_features_batch, self.gallery_features_tensor,
                self.k1, self.k2, self.lambda_value
            )
            if dist_mat is not None:
                final_scores_or_distances = dist_mat
                is_distance = True
            # else: print("Re-ranking failed or unavailable, using initial similarities.")

        # --- Determine Best Match for Each Query Feature ---
        if is_distance:
            # Lower distance is better. Handle potential NaN/Inf if reranking produces them.
            final_scores_or_distances = torch.nan_to_num(final_scores_or_distances, nan=float('inf'))
            best_match_scores, best_match_indices = torch.min(final_scores_or_distances, dim=1)
        else:
            # Higher similarity is better
            best_match_scores, best_match_indices = torch.max(final_scores_or_distances, dim=1)

        return best_match_indices, best_match_scores, initial_similarities


    def assign_ids(self, query_features_batch):
        """
        Compares query features, assigns IDs (new or existing), and updates the gallery.

        Args:
            query_features_batch (torch.Tensor): Batch of N query features (N, D).

        Returns:
            list: A list of assigned ReID IDs (int) for each query feature.
        """
        if query_features_batch is None or query_features_batch.shape[0] == 0:
            return []

        assigned_ids = [-1] * query_features_batch.shape[0] # Initialize with -1
        features_to_add = []
        ids_to_add = []
        existing_ids_set = set(self.db_ids_list)

        best_match_indices, best_match_scores, initial_similarities = self.compare_features(query_features_batch)

        # If gallery was empty during compare_features, handle adding all as new
        if best_match_indices is None:
            for i in range(query_features_batch.shape[0]):
                new_id = get_unique_id(existing_ids_set.union(set(ids_to_add)), self.next_person_id_counter)
                assigned_ids[i] = new_id
                features_to_add.append(query_features_batch[i:i+1, :])
                ids_to_add.append(new_id)
                self.next_person_id_counter = new_id + 1 # Update counter
            if features_to_add:
                 self._update_gallery(torch.cat(features_to_add, dim=0), ids_to_add)
            return assigned_ids


        # Gallery was not empty, determine matches
        num_queries = query_features_batch.shape[0]
        for i in range(num_queries):
            best_gallery_idx = best_match_indices[i].item()
            best_score = best_match_scores[i].item() # This is similarity or distance

            # Determine match based on score and threshold
            # Use initial similarity for thresholding even if re-ranked
            initial_sim_of_best = initial_similarities[i, best_gallery_idx].item()
            is_match = False
            if initial_sim_of_best > self.similarity_threshold:
                 is_match = True

            if is_match:
                assigned_id = self.db_ids_list[best_gallery_idx]
                assigned_ids[i] = assigned_id
                # print(f"Query {i} matched existing ID {assigned_id} (Score: {initial_sim_of_best:.4f})") # Optional log
                # Optional: Update feature for existing ID (e.g., averaging) - Skipped
            else:
                # No match or score below threshold -> Assign new ID
                # Need to ensure uniqueness against currently assigned IDs in this batch too
                new_id = get_unique_id(existing_ids_set.union(set(ids_to_add)), self.next_person_id_counter)
                assigned_ids[i] = new_id
                features_to_add.append(query_features_batch[i:i+1, :]) # Add the feature tensor for this new ID
                ids_to_add.append(new_id)
                self.next_person_id_counter = new_id + 1 # Update the main counter
                # print(f"Query {i} assigned new ID {new_id} (Best score: {initial_sim_of_best:.4f})") # Optional log

        # Add all new features and IDs to the gallery at once
        if features_to_add:
            self._update_gallery(torch.cat(features_to_add, dim=0), ids_to_add)

        return assigned_ids


    def get_gallery_size(self):
        """Returns the number of unique IDs (features) currently in the gallery."""
        return self.gallery_features_tensor.shape[0]