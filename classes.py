from natsort import natsorted
import os
import re
from glob import glob
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn
import networkx as nx

import sklearn.neighbors

import imageio
import cv2
import skimage
from skimage import img_as_float32, img_as_ubyte, img_as_uint
from skimage.feature import canny
from skimage.color import rgb2gray, rgb2hsv, gray2rgb, rgba2rgb
from functools import partial
from tqdm import tqdm
tqdm = partial(tqdm, position=0, leave=True)
# caching with sane defaults
from cachier import cachier
cachier = partial(cachier, pickle_reload=False, cache_dir='data/cache')

############################## Stuff for loading and rescaling the puzzle pieces nicely ################################
SIZE = (768, 1024)

DATA_PATH_PAIRS = list(zip(
    natsorted(glob(f'puzzle_corners_{SIZE[1]}x{SIZE[0]}/images-{SIZE[1]}x{SIZE[0]}/*.png')),
    natsorted(glob(f'puzzle_corners_{SIZE[1]}x{SIZE[0]}/masks-{SIZE[1]}x{SIZE[0]}/*.png')),
))
DATA_IMGS = np.array([img_as_float32(imageio.imread(img_path)) for img_path, _ in tqdm(DATA_PATH_PAIRS, 'Loading Images')])
DATA_MSKS = np.array([img_as_float32(imageio.imread(msk_path)) for _, msk_path in tqdm(DATA_PATH_PAIRS, 'Loading Masks')])

assert DATA_IMGS.shape == (48, SIZE[0], SIZE[1], 3)
assert DATA_MSKS.shape == (48, SIZE[0], SIZE[1])

with open(f'puzzle_corners_{SIZE[1]}x{SIZE[0]}/corners.json', mode='r') as f:
    DATA_CORNER_NAMES, DATA_CORNERS = json.load(f)
    DATA_CORNERS = np.array(DATA_CORNERS)

assert len(DATA_CORNER_NAMES) == len(DATA_CORNERS) == len(DATA_IMGS) == len(DATA_MSKS) == len(DATA_PATH_PAIRS)

SCALE = 0.25

MATCH_IMGS = [cv2.resize(img, None, fx=SCALE, fy=SCALE) for img in tqdm(DATA_IMGS, 'Resizing Images')]
MATCH_MSKS = [cv2.resize(img, None, fx=SCALE, fy=SCALE) for img in tqdm(DATA_MSKS, 'Resizing Masks')]
MATCH_CORNERS = DATA_CORNERS 

print('\n', DATA_IMGS[0].shape, '->', MATCH_IMGS[0].shape)

################################################ Define our three classes #############################################
class Edge:
    def __init__(self, point1, point2, contour, parent_piece):
        self.parent_piece = parent_piece # Puzzle piece the edge belongs to
        # first and last points
        self.point1 = point1  # Points should be anti-clockwise
        self.point2 = point2 
        self.connected_edge = None
        self.is_flat = None

    def info(self):
        print("Point 1: ", self.point1)
        print("Point 2: ", self.point2)

class Piece:
    def __init__(self, image, idx):
        self.piece_type = None
        self.inserted = False
        # Keep track of where the pieces corner's are. Used to construct the edge variables
        self.corners = None  # randomly ordered corners
        self.top_left = None
        self.top_right = None
        self.bottom_left = None
        self.bottom_right = None
        # Edges are anti-clockwise
        self.top_edge = None
        self.left_edge = None
        self.bottom_edge = None
        self.right_edge = None
        # Edge list used for BFS generator and in inserting function to search for the necessary edge
        self.edge_list = None
        # We hold the actual image of the piece so we can insert it onto the canvas
        self.image = image
        self.idx = idx
        # We also hold the mask and transform it with the image so we always know where our piece is in the image
        self.mask = None
        # Holds image after mapping
        self.dst = None
        # ~=~=~=~=~=~=~=~=~=~=~=~=~=~ #
        self.extract_features()
        self.classify_pixels()
        self.find_corners()
        self.find_edges()

    def return_edge(self): # generator which can be used to loop through edges in the BFS
        while True:
            for edge in self.edge_list:
                yield(edge)

    def display_im(self): # Displays puzzle piece image
        plt.imshow(self.image)
        plt.show()
        plt.close()

    def print_corners(self): # Prints the coordinates of the puzzle piece's corners
        print("Top left: ", self.top_left)
        print("top right: ", self.top_right)
        print("bottom right: ", self.bottom_right)
        print("bottom left: ", self.bottom_left)

    def print_edges(self): # Prints the information of the puzzle piece's edges
        print("Top Edge")
        self.top_edge.info()
        print("Left Edge")
        self.left_edge.info()
        print("Bottom Edge")
        self.bottom_edge.info()
        print("Right Edge")
        self.right_edge.info()

    def update_edges(self, transform):
        # Question 3: TODO: Update the corner and edge information of the puzzle piece
        column = np.zeros((4, 1)) + 1
        column = np.append(self.corners[:, ::-1], column, axis  = 1)
        corners = np.dot(column, transform.transpose())
		
		#  flip the coordinates before doing the multiplication
        self.corners = corners[:, ::-1]
        
        #Update edges
        for edge in self.edge_list[:4]:
            if not edge == None:
			# append 1 to the coordinates and right multiply by the transform
                point1 = np.append(edge.point1[::-1], 1)
                point2 = np.append(edge.point2[::-1], 1)
                
				# transpose the transform for the right multiplication to work
                point1 = np.dot(point1, transform.transpose())
                point2 = np.dot(point2, transform.transpose())
                
				# flip back the coordinates after doing the multiplication
                edge.point1 = point1[::-1]
                edge.point2 = point2[::-1]
        return

    def extract_features(self):
        # Function which will extract all the necessary features to classify pixels
        # into background and foreground
        # Should take no input and use self.image. Returns the features image (Not for Lab 7)
        return

    def classify_pixels(self):
        # Uses the feature image from self.extract_features to classify pixels
        # into foreground and background pixels. Returns the inferred mask
        # and should update self.mask with this update as we need it in future (Not for Lab 7)
        self.mask = MATCH_MSKS[self.idx]

    def find_corners(self):
        # Finds the corners of the puzzle piece (should use self.mask). Needs to update
        # the corner info of the object (eg: self.top_left). (Not for Lab 7)
        corners = MATCH_CORNERS[self.idx] * self.mask.shape[::-1]

        # sort in anti-clockwise direction
        angle_around_center = np.arctan2(*(corners - corners.mean(axis=0)).T)
        self.corners = corners[np.argsort(angle_around_center), :]

        self.top_left = self.corners[0][::-1] 
        self.top_right = self.corners[3][::-1] 
        self.bottom_right = self.corners[2][::-1] 
        self.bottom_left = self.corners[1][::-1] 
        
    def find_edges(self):
        # Finds the contour information from self.mask. Should then create the
        # edge objects for this piece. Also needs to update self.edge_list 
        # (ending in None) and self.piece_type based on number of non-straight edges (not for Lab 7)
        self.top_edge = Edge(self.top_right, self.top_left, None, self) #[0][0], [0][-1]
        self.left_edge = Edge(self.top_left, self.bottom_left, None, self) #1
        self.bottom_edge = Edge(self.bottom_left, self.bottom_right, None, self) #2
        self.right_edge = Edge(self.bottom_right, self.top_right, None, self) #3
        self.edge_list = [self.top_edge, self.left_edge, self.bottom_edge, self.right_edge, None]

    def insert(self, canvas): # Inserts the piece into the canvas using an affine transformation
        # TODO: Implement this function
        # Question 1
        count_inserted = 0
        piece_types = ['corner', 'edge', 'interior'] 
        for edge in self.edge_list[:4]:
            #Make sure the edge has a connected edge
            if not edge.connected_edge == None:
                if edge.connected_edge.parent_piece.inserted == True:
                    count_inserted += 1
        if count_inserted <= 2:
            self.piece_type = piece_types[count_inserted]
        else:
            raise Exception("INVALID PIECE TYPE")      
        
        # Question 2 (Corner insertion Case)
        pts_src, pts_dst = [],[]        
        if self.piece_type == 'corner':
            # Find two flat edges
            flat_edge_count = 0
            for edge in self.edge_list[:4]:
                if (edge.is_flat and flat_edge_count == 0):
                    first_edge = edge
                    flat_edge_count += 1
                elif(edge.is_flat):
                    second_edge = edge            
            
            # Map first_edge.point1 and second_edge.point2 
            pts_src = np.float32([ first_edge.point2[::-1], first_edge.point1[::-1], second_edge.point2[::-1]])
            pts_dst = np.float32([[0, 800], [ 0, 800 - abs(first_edge.point2[0] - first_edge.point1[0]) ], [0 + abs(second_edge.point2[1] - second_edge.point1[1]), 800]])
            
            # Use mapping with cv2.getAffineTransform(·) to get the matrix M
            M = cv2.getAffineTransform(pts_src, pts_dst)
            self.dst = cv2.warpAffine (self.image, M, (700, 800))
            self.mask = cv2.warpAffine(self.mask, M, (700, 800))
            self.update_edges(M)
			
            # “tattoo” the correct pixels of self.dst onto the canvas
            canvas.update(self.mask, self.dst)
		
		# Question 4 (Interior insertion Case)
        elif(self.piece_type == 'interior'):
            for edge in self.edge_list[:4]:
                if edge.connected_edge != None and edge.connected_edge.parent_piece.inserted == True:
                    # check if edge.point is already in pts_src (we can’t insert redundant points)
                    if not list(edge.point1[::-1])  in pts_src:
                        pts_src.append([ edge.point1[::-1][0],edge.point1[::-1][1]])
                        pts_dst.append([ edge.connected_edge.point2[::-1][0], edge.connected_edge.point2[::-1][1]])
                        
                    if not list(edge.point2[::-1])  in pts_src:
                        pts_src.append([ edge.point2[::-1][0],edge.point2[::-1][1]])
                        pts_dst.append([ edge.connected_edge.point1[::-1][0], edge.connected_edge.point1[::-1][1]])            
            
            # Get the Affine Transform M using pts_src and pts_dst
            M = cv2.getAffineTransform(np.float32(pts_src), np.float32(pts_dst) )
            self.dst = cv2.warpAffine (self.image, M, (700, 800))
            self.mask = cv2.warpAffine(self.mask, M, (700, 800 ))
            self.update_edges(M)
			# “tattoo” the correct pixels of self.dst onto the canvas
            canvas.update(self.mask, self.dst)
            
        #Question 5 (Edge insertion Case)
        elif(self.piece_type == 'edge'):
            for edge in self.edge_list[:4]:
                if edge.connected_edge != None and edge.connected_edge.parent_piece.inserted == True:
                    # check if edge.point is already in pts_src (we can’t insert redundant points)
                    if not list(edge.point1) in pts_src:
                        pts_src.append([ edge.point1[::-1][0],edge.point1[::-1][1]])
                        pts_dst.append([ edge.connected_edge.point2[::-1][0], edge.connected_edge.point2[::-1][1]])
					
                    if not list(edge.point2) in pts_src:
                        pts_src.append([ edge.point2[::-1][0],edge.point2[::-1][1]])
                        pts_dst.append([ edge.connected_edge.point1[::-1][0], edge.connected_edge.point1[::-1][1]])
            
			# calculate scaling ratio
            orig_norm = np.linalg.norm(np.array(pts_src[0]) - np.array(pts_src[1]))
            canvas_norm = np.linalg.norm(np.array(np.array(pts_dst[0])- np.array(pts_dst[1])))
            ratio = orig_norm / canvas_norm
            
            # Calculate the third canvas coordinates point for the two cases
            # Case 1: Inserting an edge piece which lies along the bottom of the puzzle
            if (pts_dst[0][0]-pts_dst[1][0]) > (pts_dst[0][1]-pts_dst[1][1]):
                for edge in self.edge_list:
                    if edge != None and edge.is_flat != None:
                        # obtain 3 canvas coordinates using edge that will lie along the bottom
                        if edge.is_flat:    
                            pts_src.append([ edge.point2[::-1][0],edge.point2[::-1][1]])
                            edge_norm = np.linalg.norm(np.array([edge.point1[::-1][0],edge.point1[::-1][1]]) - np.array([edge.point2[::-1][0], edge.point2[::-1][1]]) )
                            pts_dst.append([pts_dst[1][0]+int(ratio*edge_norm),pts_dst[1][1]])
                            break
                            
                # Get the Affine Transform M using pts_src and pts_dst
                M = cv2.getAffineTransform(np.float32(pts_src), np.float32(pts_dst) )
                self.dst = cv2.warpAffine (self.image, M, (700, 800))
                self.mask = cv2.warpAffine(self.mask, M, (700, 800))
                self.update_edges(M)
				# “tattoo” the correct pixels of self.dst onto the canvas
                canvas.update(self.mask, self.dst)
            
            # Case 2: Inserting an edge piece which lies along the left side of the puzzle 
            else:
                for edge in self.edge_list[::-1]:
                    if edge != None and edge.is_flat != None:
                        # obtain 3 canvas coordinates using edge that will lie along left side of the puzzle
                        if edge.is_flat:
                            pts_src.append([ edge.point1[::-1][0],edge.point1[::-1][1]])
                            edge_norm = np.linalg.norm(np.array([edge.point1[::-1][0],edge.point1[::-1][1]]) - np.array([edge.point2[::-1][0],edge.point2[::-1][1]]) )
                            pts_dst.append([pts_dst[0][0],pts_dst[0][1] - int(ratio*edge_norm)])
                            break
                
                # Get the Affine Transform M using pts_src and pts_dst
                M = cv2.getAffineTransform(np.float32(pts_src), np.float32(pts_dst) )
                self.dst = cv2.warpAffine (self.image, M, (700, 800))
                self.mask = cv2.warpAffine(self.mask, M, (700, 800))
                self.update_edges(M)
				# “tattoo” the correct pixels of self.dst onto the canvas
                canvas.update(self.mask, self.dst)
                
        else:
            raise Exception("Invalid piece type")
                                       
        # print("Inserting piece: ", self.idx)
	    

class Puzzle(object):
    def __init__(self, imgs):
        # generate all piece information
        self.pieces = [
            Piece(img, idx)
            for idx, img in tqdm(enumerate(imgs), 'Generating Pieces')
        ]
        self._fill_connections()

    def _fill_connections(self):
        connections = np.ones((48,4,2))*-1
        connections[0,2] = [26,1]
        connections[0,3] = [5,3]
        connections[1,0] = [14,3]
        connections[1,2] = [29,3]
        connections[1,3] = [22,2]
        connections[2,0] = [19,0]
        connections[2,1] = [12,1]
        connections[2,2] = [7,2]
        connections[2,3] = [16,0]
        connections[3,0] = [44,0]
        connections[3,3] = [6,1]
        connections[4,1] = [5,1]
        connections[4,2] = [41,0]
        connections[4,3] = [34,1]
        connections[5,0] = [7,0]
        connections[5,1] = [4,1]
        connections[5,3] = [0,3]
        connections[6,0] = [37,0]
        connections[6,1] = [3,3]
        connections[6,3] = [32,1]
        connections[7,0] = [5,0]
        connections[7,1] = [26,0]
        connections[7,2] = [2,2]
        connections[7,3] = [41,1]
        connections[8,0] = [15,0]
        connections[8,1] = [46,1]
        connections[9,0] = [25,2]
        connections[9,1] = [47,2]
        connections[9,2] = [28,0]
        connections[9,3] = [12,3]
        connections[10,0] = [33,2]
        connections[10,2] = [31,0]
        connections[10,3] = [11,1]
        connections[11,0] = [19,2]
        connections[11,1] = [10,3]
        connections[11,2] = [23,1]
        connections[11,3] = [36,3]
        connections[12,0] = [41,2]
        connections[12,1] = [2,1]
        connections[12,2] = [35,1]
        connections[12,3] = [9,3]
        connections[13,0] = [27,1]
        connections[13,1] = [22,0]
        connections[13,2] = [25,0]
        connections[13,3] = [36,1]
        connections[14,0] = [30,1]
        connections[14,1] = [15,2]
        connections[14,3] = [1,0]
        connections[15,0] = [8,0]
        connections[15,2] = [14,1]
        connections[15,3] = [40,3]
        connections[16,0] = [2,3]
        connections[16,1] = [26,3]
        connections[16,3] = [33,0]
        connections[17,0] = [43,2]
        connections[17,1] = [37,1]
        connections[17,2] = [32,0]
        connections[17,3] = [20,3]
        connections[18,1] = [34,3]
        connections[18,2] = [38,2]
        connections[18,3] = [21,1]
        connections[19,0] = [2,0]
        connections[19,1] = [33,3]
        connections[19,2] = [11,0]
        connections[19,3] = [35,2]
        connections[20,0] = [39,0]
        connections[20,1] = [40,1]
        connections[20,2] = [27,3]
        connections[20,3] = [17,3]
        connections[21,1] = [18,3]
        connections[21,2] = [24,1]
        connections[22,0] = [13,1]
        connections[22,1] = [30,2]
        connections[22,2] = [1,3]
        connections[22,3] = [45,0]
        connections[23,0] = [43,1]
        connections[23,1] = [11,2]
        connections[23,2] = [31,3]
        connections[23,3] = [37,2]
        connections[24,1] = [21,2]
        connections[24,2] = [38,1]
        connections[24,3] = [42,1]
        connections[25,0] = [13,2]
        connections[25,1] = [45,3]
        connections[25,2] = [9,0]
        connections[25,3] = [35,0]
        connections[26,0] = [7,1]
        connections[26,1] = [0,2]
        connections[26,3] = [16,1]
        connections[27,0] = [30,3]
        connections[27,1] = [13,0]
        connections[27,2] = [43,3]
        connections[27,3] = [20,2]
        connections[28,0] = [9,2]
        connections[28,1] = [38,3]
        connections[28,2] = [34,2]
        connections[28,3] = [41,3]
        connections[29,1] = [42,3]
        connections[29,2] = [45,1]
        connections[29,3] = [1,2]
        connections[30,0] = [40,0]
        connections[30,1] = [14,0]
        connections[30,2] = [22,1]
        connections[30,3] = [27,0]
        connections[31,0] = [10,2]
        connections[31,2] = [44,2]
        connections[31,3] = [23,2]
        connections[32,0] = [17,2]
        connections[32,1] = [6,3]
        connections[32,3] = [39,1]
        connections[33,0] = [16,3]
        connections[33,2] = [10,0]
        connections[33,3] = [19,1]
        connections[34,1] = [4,3]
        connections[34,2] = [28,2]
        connections[34,3] = [18,1]
        connections[35,0] = [25,3]
        connections[35,1] = [12,2]
        connections[35,2] = [19,3]
        connections[35,3] = [36,2]
        connections[36,0] = [43,0]
        connections[36,1] = [13,3]
        connections[36,2] = [35,3]
        connections[36,3] = [11,3]
        connections[37,0] = [6,0]
        connections[37,1] = [17,1]
        connections[37,2] = [23,3]
        connections[37,3] = [44,1]
        connections[38,0] = [47,1]
        connections[38,1] = [24,2]
        connections[38,2] = [18,2]
        connections[38,3] = [28,1]
        connections[39,0] = [20,0]
        connections[39,1] = [32,3]
        connections[39,3] = [46,3]
        connections[40,0] = [30,0]
        connections[40,1] = [20,1]
        connections[40,2] = [46,2]
        connections[40,3] = [15,3]
        connections[41,0] = [4,2]
        connections[41,1] = [7,3]
        connections[41,2] = [12,0]
        connections[41,3] = [28,3]
        connections[42,1] = [24,3]
        connections[42,2] = [47,0]
        connections[42,3] = [29,1]
        connections[43,0] = [36,0]
        connections[43,1] = [23,0]
        connections[43,2] = [17,0]
        connections[43,3] = [27,2]
        connections[44,0] = [3,0]
        connections[44,1] = [37,3]
        connections[44,2] = [31,2]
        connections[45,0] = [22,3]
        connections[45,1] = [29,2]
        connections[45,2] = [47,3]
        connections[45,3] = [25,1]
        connections[46,1] = [8,1]
        connections[46,2] = [40,2]
        connections[46,3] = [39,3]
        connections[47,0] = [42,2]
        connections[47,1] = [38,0]
        connections[47,2] = [9,1]
        connections[47,3] = [45,2]
        connections = connections.astype(np.int16)
        for i in range(connections.shape[0]):
            for j in range(connections.shape[1]):
                if not list(connections[i,j]) == [-1,-1]:
                    self.pieces[i].edge_list[j].connected_edge=self.pieces[connections[i,j][0]].edge_list[connections[i,j][1]]
                else:
                    self.pieces[i].edge_list[j].is_flat = True

# Create our canvas with the necessary size
class Canvas:
    
    def __init__(self):
        #initialize canvas in the puzzle object
        self.canvas = np.zeros((800,700,3))
        
    def update(self, mask, dst):
        # change our mask to have 3 channels to allow for blending with canvas
        mask = np.dstack([mask,mask,mask])
		
		# Use the mask to transfer only the puzzle piece pixels onto the canvas by blending the two
        self.canvas = mask * dst + (1 - mask)*self.canvas
    
    # display current canvas state	
    def display(self):
        plt.imshow(self.canvas)
        plt.axis(False)
        plt.show()
	
    # method to return the canvas, just in case we need to display it differently
    def get(self):
        return self.canvas

# Create our canvas with the necessary size
canvas = Canvas()        
