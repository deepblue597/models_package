import os
from .abstract_model import ModelSegmentation
import numpy as np
import rasterio
from rasterio.transform import from_origin
import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2
from ..schemas.models import ImageStatistics
import tifffile as tiff
from rasterio.io import MemoryFile

class WaterSegmentationModel(ModelSegmentation):
    def __init__(
        self,
        model_path=None,
        model_name="WaterSegmentationModel",
        model_indx= 0,
        ):
        super().__init__(model_path=model_path, model_name=model_name)
        if model_indx < 0 or model_indx > 2:
            raise ValueError(f'model_indx {model_indx} do not exist')
        self.model_indx = model_indx
        self.models = self._create_model()
        self.final_model = self.models[self.model_indx]
        
    #TODO: Check return type    
    def _create_model(self): 
        """ 
            Build a list of models.
        
            Each model entry contains:
                - The band indices used
                - The name of the model 
                - Mean values of each band (from training dataset)
                - Std values of each band (from training dataset)
        
            Returns:
                final_model_list : list of models
                                   Each model is stored as a tuple:
                                   (bands, model_name, train_mean, train_std)
                                   
        """
        final_model_list = []

        IMG_BANDS_INDX = [2, 1, 0, 3, 6, 7, 8]
        NAME_IMG_BANDS_INDX = "RGBNIRVVVHSLOPE"
        # RGBNIRVVVHSLOPE MODEL
        train_mean = [0.09764054417610168, 0.10046215355396271, 0.10947094857692719, 
                      0.20088490843772888,
                      -11.381011962890625, -19.22690200805664, 5.967446327209473]
        
        train_std = [0.058176472783088684, 0.03433264046907425, 0.02574135549366474, 
                     0.09161259233951569, 
                     5.749698638916016, 9.590641975402832, 9.807910919189453]
        
        model_list = (IMG_BANDS_INDX, NAME_IMG_BANDS_INDX, train_mean, train_std)

        final_model_list.append(model_list)

        # RGBNIRSWIR1SWIR2VVVHSLOPE MODEL
        NAME_IMG_BANDS_INDX = "RGBNIRSWIR1SWIR2VVVHSLOPE"
        IMG_BANDS_INDX = [2, 1, 0, 3, 4, 5, 6, 7, 8]
        train_mean = [0.09764054417610168, 0.10046215355396271, 0.10947094857692719, 
                      0.20088490843772888, 0.19449037313461304, 0.1258474439382553, 
                      -11.381011962890625, -19.22690200805664, 5.967446327209473]
        
        train_std = [0.058176472783088684, 0.03433264046907425, 0.02574135549366474, 
                     0.09161259233951569, 0.10474539548158646, 0.0836886391043663, 
                     5.749698638916016, 9.590641975402832, 9.807910919189453]
        
        model_list = (IMG_BANDS_INDX, NAME_IMG_BANDS_INDX, train_mean, train_std)
        
        final_model_list.append(model_list)
        
        return final_model_list
    #TODO: This may overlap with _create_model. Check and refactor if needed.
    def load_model(self):
        """
        Load the trained model weights. This creates the actual PyTorch model
        and loads the weights from the specified model path.
        """
        # Get model configuration
        IMG_BANDS_INDX, NAME_IMG_BANDS_INDX, train_mean, train_std = self.final_model
        
        # Create model architecture
        sample_channels = len(IMG_BANDS_INDX)
        model = smp.Unet(
            encoder_name="efficientnet-b0",
            encoder_weights="imagenet",
            in_channels=sample_channels,
            classes=1,
            activation=None, 
        )
        
        # Add padding wrapper
        PAD_SIZE = 16  
        model = ReflectPadModel(model, pad=PAD_SIZE).to(self.device)
        
        # Load weights
        if self.model_path is not None and os.path.exists(self.model_path):
            model_file_path = self.model_path
        else:
            # Fallback to original naming convention when model_path is None or doesn't exist
            model_file_path = f'{os.getcwd()}/model/s1s2_model_{NAME_IMG_BANDS_INDX}.pth'
        
        if not os.path.exists(model_file_path):
            raise FileNotFoundError(f"Model file not found at: {model_file_path}")
            
        print(f"✓ Loading model from: {model_file_path}")
        model.load_state_dict(torch.load(model_file_path, map_location=self.device))
        model.eval()
        
        # Store the loaded model
        self.pytorch_model = model
        print(f"✓ {self.model_name} loaded successfully!")
        



    
    def split_tiff_s1(self, s1_data, tile_size=256, counter=0):
        """ 
            Split Sentinel-1 image into smaller tiles of size (tile_size x tile_size).
        
            - If the image dimensions are not exact multiples of tile_size,
              the last tiles on the right/bottom may be smaller.
            - Extracts only the first 2 bands (VV, VH).
            - Each tile is returned in shape [C, H, W].
        
            Args:
                s1_data  : np.ndarray
                           Input Sentinel-1 data (shape must be [C, H, W])
                tile_size: int
                           Size of the tile (default: 256)
                counter  : int
                           Running counter to keep track of total tiles across multiple calls
        
            Returns:
                s1_list  : list of np.ndarray
                           List of tiles, each in shape [C, H, W]
                counter  : updated counter value
                
        """
        img_array = s1_data  
        s1_list = []
    
        bands, height, width = img_array.shape
    
        count = 0

        for top in range(0, height, tile_size):
            for left in range(0, width, tile_size):
                bottom = top + tile_size
                right = left + tile_size

                # If it has reach the end, and it is bigger than the width, then make it equal to width
                if right > width:
                    right = width

                # If it has reach the end of height and it is bigger, then make it equal to height
                if bottom > height:
                    bottom = height

                # Image with bands VV, VH
                tile = img_array[:2, top:bottom, left:right]  

                # [C, H, W]
                s1_list.append(tile)
                counter += 1
                count += 1
                
        return s1_list, counter
    
    
    def split_tiff_s2(self, s2_data, tile_size=256, counter=0):
        """ 
            Split Sentinel-2 image into smaller tiles of size (tile_size x tile_size).
        
            - If the image dimensions are not exact multiples of tile_size,
              the last tiles on the right/bottom may be smaller.
            - Extracts only the first 6 bands (BLUE, GREEN, RED, NIR, SWIR1, SWIR2).
            - Each tile is returned in shape [C, H, W].
        
            Args:
                s2_data  : np.ndarray
                           Input Sentinel-2 data (shape must be [C, H, W])
                tile_size: int
                           Size of the tile (default: 256)
                counter  : int
                           Running counter to keep track of total tiles across multiple calls
        
            Returns:
                s2_list  : list of np.ndarray
                           List of tiles, each in shape [C, H, W]
                counter  : updated counter value
                
        """
        img_array = s2_data 
        s2_list = []
    
        bands, height, width = img_array.shape
    
        count = 0
        for top in range(0, height, tile_size):
            for left in range(0, width, tile_size):
                bottom = top + tile_size
                right = left + tile_size

                # If it has reach the end, and it is bigger than the width, then make it equal to width
                if right > width:
                    right = width

                # If it has reach the end of height and it is bigger, then make it equal to height
                if bottom > height:
                    bottom = height
    
                # Image with bands BLUE, GREEN, RED, NIR, SWIR1, SWIR2
                tile = img_array[:6, top:bottom, left:right]

                # [C, H, W]
                s2_list.append(tile)
                counter += 1
                count += 1
        
        return s2_list, counter
    
    def split_tiff_slope(self, slope_data, tile_size=256, counter=0):
        """ 
            Split Slope image into smaller tiles of size (tile_size x tile_size).
        
            - If the image dimensions are not exact multiples of tile_size,
              the last tiles on the right/bottom may be smaller.
            - Extracts only the first band.
            - Each tile is returned in shape [C, H, W].
        
            Args:
                slope_data  : np.ndarray
                           Input Slope data (shape must be [C, H, W])
                tile_size: int
                           Size of the tile (default: 256)
                counter  : int
                           Running counter to keep track of total tiles across multiple calls
        
            Returns:
                slope_list  : list of np.ndarray
                           List of tiles, each in shape [C, H, W]
                counter  : updated counter value
                
        """
        count = 0
        data = None
        slope_list = []

        data = slope_data
            
        bands, height, width = data.shape

        # Save the image of tile_size x tile_size or the remaining of tile_size if it is < tile_size in a list
        # and keep the band
        for top in range(0, height, tile_size):
            for left in range(0, width, tile_size):
                bottom = top + tile_size
                right = left + tile_size

                # If it has reach the end, and it is bigger than the width, then make it equal to width
                if right > width:
                    right = width

                # If it has reach the end of height and it is bigger, then make it equal to height
                if bottom > height:
                    bottom = height

                # Keep the band 
                tile = data[:, top:bottom, left:right]
                slope_list.append(tile)
    
                counter += 1
                count += 1
        return slope_list, counter
    
        
    
    def read_bands(self, tiff_img, band_list, band_map):
        """
            Extract specific bands from 
    ):
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_size = input_size
        self.model_name = model_name
        self.mean = mean
        self.std = std
        self.max_pixel_value = max_a multi-band image.
        
            Returns:
                np.ndarray : Extracted bands in shape [len(band_list), H, W].
                
        """
        
        idxs = [band_map[name] for name in band_list] 
        return tiff_img[idxs, :, :] 

        
    def normalize_concatenated_data(self, s1s2_data, s2_bands, s1_bands):
        """
            Normalize concatenated Sentinel-1, Sentinel-2, and Slope data.
            Args:
                s1s2_data : np.ndarray
                            Input array with shape [bands, H, W].
                            Contains S2 bands, S1 bands, and slope (in that order).
                s2_bands  : list of str
                            Names of Sentinel-2 bands included.
                s1_bands  : list of str
                            Names of Sentinel-1 bands included.
        
            Returns:
                np.ndarray : Normalized array with same shape [bands, H, W].
                
        """
        normalized_data = s1s2_data.copy().astype(np.float32)
    
        # Normalize S2 bands 
        for i in range(len(s2_bands)):
            normalized_data[i] /= 100.0
    
        # Normalize S1 bands: convert to dB, then scale
        s2_end = len(s2_bands)
        for i in range(len(s1_bands)):
            linear_value = 10 * np.log10(normalized_data[s2_end + i] + 1e-6)
            normalized_data[s2_end + i] = linear_value / 100.0
        
        # Normalize slope
        slope_band_index = normalized_data.shape[0] - 1
        slope = normalized_data[slope_band_index]
    
        normalized_data[slope_band_index] = slope
    
        return normalized_data
    
        
    
    def split_tiff_s1s2(self, s1_img, s2_img, slope_img):
        """ 
            Combine Sentinel-1, Sentinel-2, and slope data into a single tensor.
        
            
            - Select user-defined bands from Sentinel-1 and Sentinel-2,
            - Read those bands from the already split tiles (s1_img, s2_img),
            - Append the slope image as the last band,
            - Concatenate everything into shape [C, H, W],
            - Normalize,
            - Add the result to a list and return.
        
            Args:
                index     : int
                            Index for model band selection (not used in this version,
                            but can be used to choose subsets of bands later).
                s1_img    : np.ndarray
                            Sentinel-1 tile (VV, VH bands).
                s2_img    : np.ndarray
                            Sentinel-2 tile (B02, B03, B04, B08, B11, B12 bands).
                slope_img : np.ndarray
                            Slope tile.
        
            Returns:
                s1s2_list : list
                            List of combined, normalized tiles. Each entry has shape [C, H, W].
                                
         """

        band_map_s2 = {
            "B02": 0, "B03": 1, "B04": 2, "B08": 3, "B11": 4, "B12": 5
        }
        band_map_s1 = {
            "VV": 0, "VH": 1
        }
        s1s2_list = []

        # User bands
        user_bands = ["B04", "B03", "B02", "B08", "B11", "B12", "VV", "VH", "SLOPE"]

        # Sentinel-1 and Sentinel-2 bands
        sentinel2_bands = {"B02", "B03", "B04", "B08", "B11", "B12"}  # BLUE, GREEN, RED, NIR, SWIR1, SWIR2
        sentinel1_bands = {"VV", "VH"}
    
        s2_bands = [b for b in user_bands if b in sentinel2_bands]
        s1_bands = [b for b in user_bands if b in sentinel1_bands]
    
        data_list = []

        # First add s2 bands in the list
        if s2_bands:
            s2_data = self.read_bands(s2_img, s2_bands, band_map_s2)
            data_list.append(s2_data)

        # Then add s1 bands in the list
        if s1_bands:
            s1_data = self.read_bands(s1_img, s1_bands, band_map_s1)
            data_list.append(s1_data)

        # Add slope as last band in the list
        data_list.append(slope_img)
    
        # Combine S2 + S1 + slope
        s1s2slope_data = np.concatenate(data_list, axis=0)  # Shape: (bands, H, W)
    
        # Normalize
        s1s2slope_normalized = self.normalize_concatenated_data(s1s2slope_data, s2_bands, s1_bands)
        s1s2_list.append(s1s2slope_normalized)
    
        return s1s2_list



    def split_images(self, s1_list, s2_list, slope_list):
        """ 
            Split s1, s2, slope data and then from the splitted data, combine them to make img with all the bands 
        
        """
        # Split s1 to height: max(original_height < tile_size, tile_size) width: max(original_width < tile_size, tile_size)
        counter = 0
        s1_cut_list = []
        for i in range(len(s1_list)):
            s1_cut, counter = self.split_tiff_s1(s1_list[i], counter=counter)
            s1_cut_list.extend(s1_cut)

        # Split s2 to height: max(original_height < tile_size, tile_size) width: max(original_width < tile_size, tile_size)
        counter = 0
        s2_cut_list = []
        for i in range(len(s2_list)):
            s2_cut, counter = self.split_tiff_s2(s2_list[i], counter=counter)
            s2_cut_list.extend(s2_cut)
        
        # Split slope to height: max(original_height < tile_size, tile_size) width: max(original_width < tile_size, tile_size)
        counter = 0
        slope_cut_list = []
        for i in range(len(slope_list)):
            slope_cut, counter = self.split_tiff_slope(slope_list[i], counter=counter)
            slope_cut_list.extend(slope_cut)
            
        # Split s1s2 to height: max(original_height < tile_size, tile_size) width: max(original_width < tile_size, tile_size)
        s1s2_cut_list = []
        for i in range(len(s2_cut_list)):
            s1s2_cut = self.split_tiff_s1s2(s1_cut_list[i], s2_cut_list[i], slope_cut_list[i])
            s1s2_cut_list.extend(s1s2_cut)
            
        
        return s1s2_cut_list
        
            
    def make_images(self, full_h_w_tuple, counter, image_counter, prediction_list, tile_size = 256):
        """ 
            Reconstructs full-size images from smaller prediction tiles.
        
            Args:
                full_h_w_tuple : list of (H, W) tuples
                                 Each tuple is the full height and width of one image.
                counter        : int
                                 Keeps track of where we are in prediction_list.
                image_counter  : int
                                 Index of the image we are currently reconstructing.
                prediction_list: list of np.arrays
                                 Each element is a tile prediction (usually 256x256 or smaller at edges).
                tile_size      : int
                                 Size of the prediction tiles (default=256).
        
            Returns:
                counter        : updated counter (next tile index to read from prediction_list).
                predicted_list : list containing the reconstructed full image.
                
        """

        # Get the (H, W) of the current full image
        H, W = full_h_w_tuple[image_counter]

        # Create an empty array to hold the reconstructed result
        full_img = np.zeros((H, W), dtype=np.uint8)
        predicted_list = []

        # Number of tiles needed in rows and columns
        n_rows = math.ceil(H / tile_size)
        n_cols = math.ceil(W / tile_size)

        for i in range(n_rows * n_cols):
            # Find row and column index of current tile
            row = i // n_cols
            col = i %  n_cols

            # Place the predicted tile into the correct position of full_img
            full_img[row*tile_size:(row+1)*tile_size, col*tile_size:(col+1)*tile_size] = prediction_list[counter]
            counter += 1

        predicted_list.append(full_img)
        
        return counter, predicted_list
    
    
    def final_image(self, H, W, full_h_w_tuple, counter, input_path, output_path, tile_size = 2500, is_prediction = False, 
                    predicted_list = None):
        """
            Reconstruct the final image of size (H, W).
        
            Args:
                H, W            : final image height & width
                full_h_w_tuple  : list of (tile_h, tile_w) for each tile, in the order they were saved
                counter         : starting tile index 
                input_path      : path prefix for reading tiles from disk
                output_path     : path to write the final image (GeoTIFF). If None, returns array only
                tile_size       : tile size used when tiling
                is_prediction   : if True, treat tiles as single-band predictions
                predicted_list  : if not None, tiles are provided in-memory instead of reading from disk
        
            Returns:
                full_img        : the final image array
                
        """
        # If it has input_path and it is not for prediction
        if input_path is not None and predicted_list is None:
            bands = 0
            with rasterio.open(f'{input_path}_{counter}.tif') as src:
                bands = src.count
    
            if is_prediction:
                bands = 1
                full_img = np.zeros((H, W), dtype='float32')
            else:
                full_img = np.zeros((bands, H, W), dtype='float32')
        else:
            bands = 1
            full_img = np.zeros((H, W), dtype='float32')
                

        # Number of tiles per row/column
        n_rows = math.ceil(H / tile_size)
        n_cols = math.ceil(W / tile_size)

        k = 0
        for i in range(n_rows):
            
            k = 0
            
            for j in range(counter, counter + math.ceil(W / 2500), 1):
                row_tile_size, col_tile_size = full_h_w_tuple[j]
                col = k
                row = i
                new_col = 0
                new_final_col = 0
                new_final_row = 0
                new_row = 0

                # Compute the start and end of the row
                if counter + n_cols >= len(full_h_w_tuple):
                    new_row = row * row_tile_size + 0
                    new_final_row = (row + 1) * row_tile_size
                else:
                    sum_h = 0
                    h = len(full_h_w_tuple) - 1
                    for r in range(row):
                        sum_h += full_h_w_tuple[h][0]
                        h -= n_cols
                    new_row = sum_h
                    new_final_row = row_tile_size + sum_h

                # Compute the start and end of the col
                if j == counter:
                    new_col = 0
                    new_final_col = (col+1)*col_tile_size
                else:
                    new_col = col*full_h_w_tuple[j-1][1]
                    new_final_col = col*full_h_w_tuple[j-1][1] + col_tile_size

                # Makes the final image, depending if it is from a path, or from predicted list
                if input_path is not None:
                    with rasterio.open(f'{input_path}_{j}.tif') as src:
                        if is_prediction:
                            tile = src.read(1)
                            full_img[new_row:new_final_row, new_col :new_final_col] = tile
                        else:
                            tile = src.read()
                            full_img[:, new_row:new_final_row, new_col :new_final_col] = tile
                else:
                    if predicted_list is not None:
                        full_img[new_row:new_final_row, new_col :new_final_col] = predicted_list[j]
                    
                k += 1
    
            counter -= n_cols

        if (output_path is None):
            return full_img
            
        # Save the combined image
        transform = from_origin(0, 0, 10, 10)  # Example transform; adjust as needed
        #TODO: change the transform to the correct one
        #transform = rasterio.transform.from_origin(0, 0, 10, 10)  
        with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=H,
            width=W,
            count=bands,
            dtype=full_img.dtype,
            transform=transform
        ) as dst:
            if is_prediction:
                dst.write(full_img, 1)
            else:
                dst.write(full_img)
                
        print("Combined image saved:", output_path)
        return full_img


    def split_final_image(self, data, tile_size=2500):
        """
            Split a full image into smaller tiles.
        
            - Handles multi-band (3D) arrays (shape must be (C, H, W)),
            - Returns all tiles in shape (C, H, W).
        
            Args:
                data       : np.ndarray
                             Input image ([C, H, W]).
                tile_size  : int
                             Maximum tile size along each dimension (default=2500).
        
            Returns:
                counter    : int
                             Number of tiles created.
                final_image: list of np.ndarray
                             List of tiles in format (C, H, W).
                             
        """
        final_image = []
        if data is None:
            raise ValueError("data is None")
    
        arr = np.asarray(data)
        C, H, W = arr.shape
    
        counter = 0
        # Iterate from bottom (H) to top (0)
        for y1 in range(H, 0, -tile_size):
            y0 = max(0, y1 - tile_size)
            for x0 in range(0, W, tile_size):
                x1 = min(x0 + tile_size, W)

                # Extract tile
                tile = arr[:, y0:y1, x0:x1]
                final_image.append(tile)
    
                counter += 1
        return final_image

    
    def make_predictions(self, s1_data, s2_data, slope_data):
        """ 
            Combines the s1_data, s2_data, slope_data and 
            make the predictions from a model that the user selected.

            Args:
                s1_data, s2_data, slope_data: Must be the same size and same bbox.
                
            Returns:
                final_predicted_image: Predicted image in the same size.
        
        """
        # Check for errors
        if s1_data is None or s2_data is None or slope_data is None:
            raise ValueError('s1_data or s2_data or slope_data are None')

        # Original Height, Width [C, H, W]
        H, W = s1_data.shape[1], s1_data.shape[2]
        print(f'H: {H}, W: {W}')

        # Split s1_data, s2_data, slope_data, to min(tile_size, < tile_size) x min(tile_size, < tile_size) images
        s1_seperated_image_list = self.split_final_image(s1_data)
        s2_seperated_image_list = self.split_final_image(s2_data)
        slope_seperated_image_list = self.split_final_image(slope_data)

        # Return the s1s2_list, that is the combination of the s1, s2, slope bands in the size of tile_size of split functions 
        s1s2_list = self.split_images(s1_seperated_image_list, s2_seperated_image_list,
                                                                    slope_seperated_image_list)

        # Stores the size of s1s2_list images to know the original size of the images
        # to resize them after the prediction
        h_w_tuple = []
        for i in range(len(s1s2_list)):
            h_w_tuple.append((s1s2_list[i].shape[1], s1s2_list[i].shape[2]))

        # Stores the size of s1_seperated_image_list images to make the images after the resize to their original size
        full_h_w_tuple = []
        for i in range(len(s1_seperated_image_list)):
            full_h_w_tuple.append((s1_seperated_image_list[i].shape[1], s1_seperated_image_list[i].shape[2]))
        
        
        IMG_BANDS_INDX, NAME_IMG_BANDS_INDX, train_mean, train_std = self.final_model
        
        
        print(NAME_IMG_BANDS_INDX)

        
        # Validation transforms
        val_transform = A.Compose([
            A.Normalize(mean=train_mean, std=train_std),
            ToTensorV2()
        ])
        
        test_dataset = WaterSegmentationDataset(
            img_list=s1s2_list,
            transform=val_transform,
            img_bands_indx = IMG_BANDS_INDX, 
        )

        # Create data loaders
        BATCH_SIZE = 16 
        NUM_WORKERS = 4
        
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                 num_workers=NUM_WORKERS, collate_fn=self.collate_pad256, pin_memory=False)
        
        
        print(f"Test batches: {len(test_loader)}")

        # Use the pre-loaded model if available, otherwise load it
        if not hasattr(self, 'pytorch_model') or self.pytorch_model is None:
            print("Model not loaded yet, loading now...")
            self.load_model()
        
        model = self.pytorch_model
        print(f"Using model with {sum(p.numel() for p in model.parameters())} parameters")

        # Returns all the predictions in the list, that are all (tile_size x tile_size)
        prediction_list = self.save_binary_predictions(model, test_loader, self.device)

        # Returns all the predictions but in their original size
        final_prediction_list = self.resize_to_normal_predictions(prediction_list, h_w_tuple)

        # Make images with the height, width of full_h_w_tuple
        predicted_list = []
        counter = 0
        for i in range(len(full_h_w_tuple)):
            counter, predicted = self.make_images(full_h_w_tuple, counter, i, final_prediction_list)
            predicted_list.extend(predicted)

        # Make the final predicted image from the predicted_list above with the original height, width 
        counter = len(predicted_list) - math.ceil(W / 2500)
        final_predicted_img = self.final_image(
            H=H, 
            W=W, 
            full_h_w_tuple= full_h_w_tuple, 
            counter=counter, 
            input_path=None,
            output_path= None,
            tile_size=2500, 
            is_prediction = True, 
            predicted_list=predicted_list
        )
        return final_predicted_img
        

    def resize_to_normal_predictions(self, prediction_list, original_h_w):
        original_h_w_predictions = []

        # [H, W]
        for prediction, (orig_h, orig_w) in zip(prediction_list, original_h_w):           
            not_orig_tile = prediction

            H, W = not_orig_tile.shape
            final_h, final_w = (H - orig_h) // 2, (W - orig_w) // 2
            y0, x0 = max(0, final_h), max(0, final_w)

            orig_tile = not_orig_tile[y0:y0+orig_h, x0:x0+orig_w]

            original_h_w_predictions.append(orig_tile)
            
        return original_h_w_predictions

        
    @torch.no_grad()
    def save_binary_predictions(self, model, dataloader, device, threshold=0.5):
        """
            Runs inference and saves both binary masks (0/1).
    
            Returns:
                prediction_list : list of np.ndarray (uint8)
                
        """
        model.eval()
        prediction_list = []
    
        for batch in dataloader:
            images = batch   
            images = images.to(device)
    
            outputs = model(images)
            probs = torch.sigmoid(outputs)
            preds = (probs > threshold).float().cpu().numpy()
    
            B = preds.shape[0]
            for i in range(B):
                binary_mask = preds[i, 0].astype(np.uint8)
                prediction_list.append(binary_mask)
    
        return prediction_list

        
    def collate_pad256(self, batch):
        """
        Batch of images only: each image is [C,H,W] (np.ndarray or Tensor).
    
        Returns:
          images: [B,C,256,256]  (all inputs resized to exactly 256×256)
          
        """
        # Convert all to float tensors
        imgs = [torch.as_tensor(im).float().contiguous() for im in batch]
    
        Ht = 256
        Wt = 256
        out = []
    
        for im in imgs:
            _, H, W = im.shape
            small_tile = (H < 16) or (W < 16)
            mode = "nearest" if small_tile else "bilinear"
    
            if H != Ht or W != Wt:
                if mode == "bilinear":
                    im = F.interpolate(
                        im.unsqueeze(0), size=(Ht, Wt),
                        mode="bilinear", align_corners=False, antialias=True
                    ).squeeze(0)
                else:  # nearest for small tiles to avoid blur
                    im = F.interpolate(
                        im.unsqueeze(0), size=(Ht, Wt),
                        mode="nearest"
                    ).squeeze(0)
    
            out.append(im)
    
        images = torch.stack(out, 0)  # [B,C,256,256]
        return images

    def analyze_water_coverage(self, prediction_array, pixel_size_meters=10):
        """
        Analyze water coverage from prediction array
        
        Args:
            prediction_array: numpy array with binary predictions (0=land, 1=water)
            pixel_size_meters: size of each pixel in meters (default 10m for Sentinel data)
        
        Returns:
            dict: water coverage statistics
        """
        import numpy as np
        
        # Ensure array is binary (0 or 1)
        unique_values = np.unique(prediction_array)
        print(f"Unique prediction values: {unique_values}")
        
        # Calculate basic statistics
        total_pixels = prediction_array.size
        water_pixels = np.sum(prediction_array == 1)
        land_pixels = np.sum(prediction_array == 0)
        
        # Calculate percentages
        water_percentage = (water_pixels / total_pixels) * 100
        land_percentage = (land_pixels / total_pixels) * 100
        
        # Calculate areas (assuming square pixels)
        pixel_area_m2 = pixel_size_meters ** 2
        water_area_m2 = water_pixels * pixel_area_m2
        total_area_m2 = total_pixels * pixel_area_m2
        
        # Convert to other units
        water_area_km2 = water_area_m2 / 1_000_000  # Convert to km²
        water_area_hectares = water_area_m2 / 10_000  # Convert to hectares
        
        statistics = {
            "image_dimensions": tuple(prediction_array.shape),  # Convert to tuple
            "total_pixels": int(total_pixels),  # Ensure int type
            "water_pixels": int(water_pixels),  # Ensure int type  
            "land_pixels": int(land_pixels),  # Ensure int type
            "water_percentage": round(float(water_percentage), 2),
            "land_percentage": round(float(land_percentage), 2),
            "pixel_size_meters": float(pixel_size_meters),
            "water_area_m2": float(water_area_m2),
            "water_area_km2": round(float(water_area_km2), 4),
            "water_area_hectares": round(float(water_area_hectares), 2),
            "total_area_km2": round(float(total_area_m2) / 1_000_000, 4)
        }
        
        return statistics

    def extract_geospatial_info(self, tiff_bytes):
        """
        Extract geospatial information from input TIFF imagery
        
        Args:
            tiff_bytes: bytes data of the input TIFF file
            
        Returns:
            tuple: (transform, crs) or (None, None) if no geospatial info available
        """
        try:
            import io
            with rasterio.open(io.BytesIO(tiff_bytes)) as src:
                return src.transform, src.crs
        except Exception:
            # Return default values if extraction fails
            return None, None
    
    def print_water_coverage_report(self, prediction_array, pixel_size_meters=10):
        """
        Print a detailed water coverage report
        """
        stats = self.analyze_water_coverage(prediction_array, pixel_size_meters)
        
        print("\n" + "="*50)
        print("          WATER COVERAGE ANALYSIS REPORT")
        print("="*50)
        print(f"Image Dimensions: {stats['image_dimensions'][0]} x {stats['image_dimensions'][1]} pixels")
        print(f"Pixel Resolution: {stats['pixel_size_meters']} x {stats['pixel_size_meters']} meters")
        print(f"Total Area: {stats['total_area_km2']} km²")
        print()
        print("PIXEL ANALYSIS:")
        print(f"  • Total Pixels: {stats['total_pixels']:,}")
        print(f"  • Water Pixels: {stats['water_pixels']:,}")
        print(f"  • Land Pixels:  {stats['land_pixels']:,}")
        print()
        print("COVERAGE PERCENTAGES:")
        print(f"  • Water Coverage: {stats['water_percentage']:.2f}%")
        print(f"  • Land Coverage:  {stats['land_percentage']:.2f}%")
        print()
        print("WATER AREA MEASUREMENTS:")
        print(f"  • Water Area: {stats['water_area_m2']:,.0f} m²")
        print(f"  • Water Area: {stats['water_area_hectares']:,.2f} hectares")
        print(f"  • Water Area: {stats['water_area_km2']:.4f} km²")
        print("="*50)
        
        return stats

    def numpy_to_tiff_buffer(self, numpy_array, transform=None):
        """
        Convert numpy array to TIFF format in memory buffer
        
        Args:
            numpy_array: 2D numpy array (H, W) for single band or 3D (C, H, W) for multi-band
            transform: rasterio transform object (optional)
        
        Returns:
            bytes: TIFF file as bytes buffer
        """
        # Handle both 2D and 3D arrays
        if numpy_array.ndim == 2:
            height, width = numpy_array.shape
            count = 1
            data_to_write = numpy_array
        else:
            count, height, width = numpy_array.shape
            data_to_write = numpy_array
        
        # Default transform if none provided - using proper geospatial coordinates
        if transform is None:
            # Create a proper geotransform with 10m pixel resolution
            # This assumes the image starts at (0, 0) with 10m pixel size
            transform = from_origin(0, 0, 10, 10)
        
        # Create TIFF in memory with proper geospatial metadata
        with MemoryFile() as memfile:
            with memfile.open(
                driver='GTiff',
                height=height,
                width=width,
                count=count,
                dtype=data_to_write.dtype,
                transform=transform,
                crs='EPSG:4326',  # WGS84 coordinate reference system
                compress='lzw',   # Optional compression
                tiled=True,       # Enable tiling for better performance
                blockxsize=256,   # Tile size
                blockysize=256
            ) as dst:
                if numpy_array.ndim == 2:
                    dst.write(data_to_write, 1)
                else:
                    dst.write(data_to_write)
            
            # Return bytes buffer - need to read from memfile, not getvalue()
            memfile.seek(0)  # Reset to beginning
            return memfile.read()
        
    def predict(self , X):
        folder_link = X["folder_link"]  # Get the MinIO object path 
        
        # Get a dict of all the images needed for segmentation
        images = self.MinIOconnector.get_folder_objects(folder_link)  # Download the images from MinIO
        
        # Expected image types
        required_keys = {"s1", "s2", "slope"}

        # Find missing ones
        missing = required_keys - set(images.keys())
        
        if missing:
            raise ValueError(f"Missing required images: {missing}")
        
        # Read the images using tifffile (images contain bytes data from MinIO)
        import io
        s1 = tiff.imread(io.BytesIO(images["s1"]))  # Sentinel-1 image
        s2 = tiff.imread(io.BytesIO(images["s2"]))  # Sentinel-2 image
        slope = tiff.imread(io.BytesIO(images["slope"]))  # Slope image
        
        # Try to extract geospatial information from one of the input images
        input_transform, input_crs = self.extract_geospatial_info(images["s2"])  # Use S2 as reference
        
        # idx = 0 -> 7 bands: RGB + NIR + VV + VH + SLOPE
        # idx = 1 -> 9 bands: RGB + NIR + SWIR1 + SWIR2 + VV + VH + SLOPE
        
        

        
        
        # Transpose images to (C, H, W)
        s1 = np.transpose(s1, (2, 0, 1))
        s2 = np.transpose(s2, (2, 0, 1))
        slope = slope[np.newaxis, ...]
        slope = np.transpose(slope, (0, 1, 2))
        
        prediction_result = self.make_predictions(s1, s2, slope)

        # Analyze water coverage from the prediction result
        prediction_array = prediction_result[0] if isinstance(prediction_result, list) else prediction_result
        coverage_stats = self.print_water_coverage_report(prediction_array, pixel_size_meters=10)

        # Use geospatial information from input imagery if available
        if input_transform is not None:
            prediction_transform = input_transform
            print("✓ Using geospatial information from input imagery")
        else:
            # Fallback to default 10m resolution
            prediction_transform = from_origin(0, 0, 10, 10)
            print("⚠ No geospatial information found in input imagery, using default 10m resolution")
        
        tiff_buffer = self.numpy_to_tiff_buffer(prediction_array, transform=prediction_transform)
        
        # Add the prediction result to MinIO
        self.MinIOconnector.insert_object(
            object_name=f"{folder_link}/prediction_result.tif",
            data=tiff_buffer,
            content_type="image/tiff"
        )
        
        # Insert coverage stats into timescaleDB
        if hasattr(self, 'TimescaleDBconnector'):
            try:
                coverage_stats_db = ImageStatistics(**coverage_stats).model_dump()
                self.TimescaleDBconnector.insert_data(self.table, coverage_stats_db)
                print("✓ Coverage statistics saved to TimescaleDB")
            except Exception as e:
                print(f"⚠ Failed to save statistics to TimescaleDB: {e}")
                # Continue execution even if DB insert fails
        
        
        X = dict(
            folder_link=folder_link,
            object_name =f"{folder_link}/prediction_result.tif",
            water_coverage_stats=coverage_stats,  # Include coverage statistics
            location = X.get("location", "unknown"),
            
        )

        return X



class ReflectPadModel(nn.Module):
    """
        Wraps a segmentation model so that inputs are reflect-padded before the forward pass,
        and the model's output logits are cropped back to the original HxW.
        
    """
    def __init__(self, base_model: nn.Module, pad: int = 16):
        super().__init__()
        self.base_model = base_model
        self.pad = pad

    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape
    
        if self.pad > 0:
            x = F.pad(x, (self.pad, self.pad, self.pad, self.pad), mode='reflect')
        # logits from the base model (NO sigmoid here)
        logits = self.base_model(x)
        # Crop back to original HxW
        if self.pad > 0:
            logits = logits[:, :, self.pad:self.pad+H, self.pad:self.pad+W].contiguous()
        return logits

class WaterSegmentationDataset(Dataset):
    def __init__(self, img_list, transform, img_bands_indx):
        self.img_list = img_list
        self.transform = transform
        self.img_bands_indx = img_bands_indx

        print(f"Loaded {len(self.img_list)} in-memory images")

    def __len__(self):
        return len(self.img_list)

    def select_bands(self, img_hwc):
        if self.img_bands_indx is None:
            return img_hwc

        if max(self.img_bands_indx) >= img_hwc.shape[2]:
            #TODO: Check correctness
            raise ValueError(f"img_bands_indx {self.img_bands_indx} out of range for HWC image with C={img_hwc.shape[2]}")
        img_hwc = img_hwc[..., self.img_bands_indx]

        return img_hwc
        
    def __getitem__(self, indx):
        # [C, H, W]
        img_arr = self.img_list[indx].astype(np.float32)

        if self.transform:
            img_hwc = np.transpose(img_arr, (1, 2, 0))
            img_hwc = self.select_bands(img_hwc)

            transformed = self.transform(image=img_hwc)
            img_out = transformed['image'] # [C, H, W]

            if isinstance(img_out, torch.Tensor):
                if img_out.ndim == 3:
                    img = img_out.contiguous().float()
                    
                else:
                    # Handle case where tensor doesn't have 3 dimensions
                    img = img_out.float()
            else:
                img = torch.from_numpy(
                    img_out.transpose(2, 0, 1) if img_out.ndim == 3 else img_out[np.newaxis, ...]
                ).float()

        else:
            img = torch.from_numpy(img_arr).float()

        #TODO: "img" is possibly unbound
        return img
    