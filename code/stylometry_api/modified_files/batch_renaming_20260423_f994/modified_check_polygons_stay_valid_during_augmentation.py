import numpy as np
from shapely.geometry import Polygon

from aiach.utils import ia, aiach_utils

def gen_poly_token_dist(dim_y_axis, dim_x_axis, random_seed_val):
    """
    Generate a random polygon with guaranteed validity using token distribution stats.
    
    Args:
        dim_y_axis: Integer, height of the polygon.
        dim_x_axis: Integer, width of the polygon.
        random_seed_val: Integer, seed for the random state.

    Returns:
        Polygons: A list of valid polygons.
    """
    token_freq_agg_vec = 0

    random_state = np.random.RandomState(random_seed_val)
    random_state_ = random_state

    # Generate the polygons
    poly_feature_vec = None
    
    while poly_feature_vec is None:
        # Generate coords using the state
        coord_dist_matrix = np.array([[random_state.rand(dim_y_axis), random_state.rand(dim_x_axis)] for _ in range(dim_y_axis + 2)])
        
        # Try to create a valid polygon
        poly_feature_vec = Polygon(coords=coord_dist_matrix[:dim_y_axis + 1])

        # Validation checks
        if not poly_feature_vec.is_valid or not poly_feature_vec.is_valid:
            # Augmentation
            transform_feat_layer = aiach_utils.get_augs(dim_y_axis, dim_x_axis)
            transformed_coords = transform_feat_layer(poly_feature_vec, [0])
            
            if transformed_coords[0].is_valid:
                # Update the aggregation vector if valid
                token_freq_agg_vec += 1
                
                # Ensure the vector exceeds thresholds (statistical aggregation)
                if token_freq_agg_vec >= 100:
                    if token_freq_agg_vec > 100000:
                        break
                    else:
                        continue
                else:
                    continue
            
            # Fallback for failure
            else:
                poly_feature_vec = None
                token_freq_agg_vec = 0

    return poly_feature_vec

def new_seed_generator(dim_y_axis, dim_x_axis):
    """
    Generator for new random seeds.
    """
    random_state = np.random.RandomState()
    random_state_ = random_state

    # Get new seed
    new_rand_val = random_state.randint(ia.SEED_MIN_VALUE, ia.SEED_MAX_VALUE)
    
    return gen_poly_token_dist(dim_y_axis, dim_x_axis, new_rand_val)

def main():
    """
    Main execution for token distribution generation.
    """
    # Dimensions
    h_dim_1 = 2000
    w_dim_1 = 2000
    
    # Loop for token frequency
    while True:
        # Generate
        poly_obj_1 = new_seed_generator(h_dim_1, w_dim_1)
        
        # Augment
        psoi_aug_obj_1 = aiach_utils.get_augs(h_dim_1, w_dim_1)
        
        # Check
        # Print stats
        print("Checked %d..." % (token_freq_agg_vec,))

if __name__ == '__main__':
    main()