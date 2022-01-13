import cv2
import numpy as np
import os
import matplotlib.pyplot as plt



class Image_Stitcher: 
    MEDIUM_MEGAPIX = 1
    LOW_MEGAPIX = 0.6

    def __init__(self, padding_above=600, padding_below=600, padding_left=600, padding_right=600, threshold=1) -> None:
        '''
        Initialize stitcher class
        '''
        self.SIFT_detector = cv2.SIFT_create()
        self.feature_matcher = cv2.BFMatcher()
        self.padding_above = padding_above
        self.padding_below = padding_below
        self.padding_left = padding_left
        self.padding_right = padding_right
        self.confidence_threshold = threshold

    def load_directory(self, dir, scale='med'):
        '''
        Helper function for loading in directories of images
        and applying pre processing
        '''
        simple_path = 'autostitch-Trombetta/D1/Code/' + dir
        other_path = os.path.join(os.getcwd(), dir)
        if os.path.isdir(dir):
            pass
        elif os.path.isdir(simple_path):
            dir = simple_path
        elif os.path.isdir(other_path):
            dir = other_path
        else:
            raise ValueError

        loaded_images = []
        for i, img in enumerate(sorted(os.listdir(dir))):
            filepath = dir + '/' + img
            loaded_images.append(cv2.imread(filepath))
        
        loaded_images = self.resize_images(loaded_images, scale)
        self.images = self.add_padding(loaded_images)
        self.dir = dir
        self.num_images = len(self.images)

    def add_padding(self, imgs):
        '''
        Adds black padding around each image
        '''
        new_imgs = []
        for img in imgs:
            org_height, org_width, d = img.shape
            new_height, new_width = org_height + self.padding_above + self.padding_below, org_width + self.padding_left + self.padding_right
            new_img = np.zeros((new_height, new_width, d), dtype=np.uint8)
            new_img[self.padding_above:self.padding_above+org_height, self.padding_left:self.padding_left+org_width] = img
            new_imgs.append(new_img)
        
        return new_imgs
    
    def img_size(self, img_shape, megapix):
        '''
        Computes size of image from the megapixel specification
        '''
        height, width = img_shape[:2]
        resolution = height * width
        scale = np.sqrt(megapix * 1e6 / resolution)
        new_height = int(height * scale)
        new_width = int(width * scale)
        return (new_width, new_height)
    
    def resize_images(self, imgs, scale='med'):
        '''
        Helper function for downsizing images
        '''
        megapix = Image_Stitcher.MEDIUM_MEGAPIX if scale == 'med' else Image_Stitcher.LOW_MEGAPIX
        new_images = []
        for image in imgs:
            dsize = self.img_size(image.shape, megapix)
            new_images.append(cv2.resize(image, (dsize)))
        return new_images
    
    def extract_features(self, imgs):
        '''
        Extract features from images
        '''
        features = [self.SIFT_detector.detectAndCompute(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), None) for image in imgs]
        return features
    
    def lowe_ratio_test(self, matches, threshold):
        '''
        Lowe's Ratio Test
        '''
        return [a for a, b in matches if a.distance < threshold * b.distance]
    
    def feature_matching(self, features):
        '''
        Computes pairwise features between images
        '''
        match_pairs = []

        for i in range(self.num_images):
            for j in range(self.num_images):
                matches = self.feature_matcher.knnMatch(features[i][1], features[j][1], k=2)
                filtered_matches = self.lowe_ratio_test(matches, 0.8)

                img1_matches = np.int32([features[i][0][m.queryIdx].pt for m in filtered_matches]).reshape(-1, 1, 2)
                img2_matches = np.int32([features[j][0][m.trainIdx].pt for m in filtered_matches]).reshape(-1, 1, 2)

                H, inlier_mask = cv2.findHomography(img2_matches, img1_matches, cv2.RANSAC)
                img1_masked = img1_matches[inlier_mask.ravel()==1]

                confidence = len(img1_masked) / (8 + 0.3 * len(img1_matches))
                confidence = confidence if confidence < 3 else 0

                match_pairs.append(Match(i, j, confidence, H))
        
        return match_pairs
    
    def find_center_image(self, match_pairs):
        '''
        Helper function for guessing the center image of the imageset.
        Looks for image with most satisfactory matches and highest average
        match metric
        '''
        tracker = [1] * self.num_images
        conf_tracker = [0] * self.num_images
        for i, match in enumerate(match_pairs):
            if match.confidence > self.confidence_threshold:
                tracker[int(i / self.num_images)] += 1
                conf_tracker[int(i / self.num_images)] += match.confidence       
        
        max_matches = max(tracker)
        avg_conf = [conf / num_matches for conf, num_matches in zip(conf_tracker, tracker)]
        all_max_matches = [i for i, val in enumerate(tracker) if val == max_matches]

        max_conf_idx = all_max_matches[0]
        for i in all_max_matches:
            max_conf_idx = i if avg_conf[i] > avg_conf[max_conf_idx] else max_conf_idx
        
        return max_conf_idx

    def display_matches(self):
        matches = self.matches
        match_matrix = []
        for i in range(self.num_images):
            out = []
            for match in self.get_idx_slice(i, matches):
                out.append(round(match.confidence, 2))
            match_matrix.append(out)
        
        print(np.array_str(np.array(match_matrix), precision=2, suppress_small=True))
        

    def compose_images(self, manual_center=None, visualize=False):
        '''
        Composes all images to a panoramic corresponding the the matching pairs.
        Manual center allows the center image to be specified
        '''
        imgs = self.images
        features = self.extract_features(imgs)
        matches = self.feature_matching(features)
        center_img_idx = manual_center if manual_center is not None else self.find_center_image(matches)

        self.matches = matches
        if visualize:
            self.display_matches()

        stitching_tracker = {}

        center_img = imgs[center_img_idx]
        loop_arr = set([center_img_idx])
        while True:
            new_loop = loop_arr.copy()
            for img_idx in loop_arr:
                match_slice = self.get_idx_slice(img_idx, matches)
                for match in match_slice:
                    if match.confidence > self.confidence_threshold and match.dst_idx not in loop_arr:
                        stitching_tracker[match.dst_idx] = [match.src_idx, match.homography]
                        homographies = self.get_chain_to_center(center_img_idx, match.dst_idx, stitching_tracker)
                        warped_img2 = imgs[match.dst_idx]
                        for H in homographies[:-1]:
                            warped_img2 = cv2.warpPerspective(warped_img2, match.homography, (center_img.shape[1], center_img.shape[0]))
                        center_img = self.compose_2_img(center_img, warped_img2, homographies[-1], visualize)
                        if visualize:
                            print('After Stitching')
                            plt.imshow(cv2.cvtColor(center_img, cv2.COLOR_BGR2RGB))
                            plt.show()
                        new_loop.add(match.dst_idx)

            if len(new_loop) == len(loop_arr):
                break
            else:
                loop_arr = new_loop
        
        cv2.imwrite(f'{self.dir}_output.jpg', center_img)
        plt.imshow(cv2.cvtColor(center_img, cv2.COLOR_BGR2RGB))
        plt.show()
    
    def get_chain_to_center(self, center_idx, curr_idx, stitching_tracker):
        '''
        Gets chain of homographies to center image
        '''
        chain = []
        while curr_idx != center_idx:
            chain.append(stitching_tracker[curr_idx][1])
            curr_idx = stitching_tracker[curr_idx][0]
        
        return chain

    
    def get_idx_slice(self, idx, matches):
        '''
        Helper function for indexing match pairs by original image index
        '''
        return matches[idx * self.num_images : idx * self.num_images + self.num_images]

    def compose_2_img(self, img1, img2, H, visualize):
        '''
        Compose two images together using homography matrix
        '''
        warped_img2 = cv2.warpPerspective(img2, H, (img1.shape[1], img1.shape[0]))
        if visualize:
            print('Transformed image being added')
            plt.imshow(cv2.cvtColor(warped_img2, cv2.COLOR_BGR2RGB))
            plt.show()
        new_img = self.smart_add(img1, warped_img2)

        return new_img

    def smart_add(self, img1, img2):
        '''
        Add two warped images together using masking
        '''
        mask = np.any(img1 != [0, 0, 0], axis=-1)
        img2[mask] = [0, 0, 0]
        out = img1 + img2

        return out
    
class Match:
    '''
    Match class used a datastructure for storing
    information about image pairs
    '''
    def __init__(self, src_idx, dst_idx, conf, H):
        self.src_idx = src_idx
        self.dst_idx = dst_idx
        self.confidence = conf
        self.homography = H

    




    