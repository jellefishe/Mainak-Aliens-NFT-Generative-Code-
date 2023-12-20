"""
Starter code for EECS 442 W22 HW1
"""
from pathlib import Path
import os
from pickletools import uint8
import shutil
#import cv2
import numpy as np
#import matplotlib; matplotlib.use('agg')
#import matplotlib.pyplot as plt
#from util import generate_gif, renderCube
import glob
#from skimage.io import imread
#from skimage.color import rgb2lab, lab2rgb
import moviepy.video.io.ImageSequenceClip
import moviepy.editor as mpe
import json
import math

from PIL import GifImagePlugin

from PIL import Image


# def rotX(theta):
#     """
#     Generate 3D rotation matrix about X-axis
#     Input:  theta: rotation angle about X-axis
#     Output: Rotation matrix (3 x 3 array)
#     """
#     return [[1, 0, 0],[0, np.cos(theta), -1*(np.sin(theta))],[0, np.sin(theta), np.cos(theta)]]


# def rotY(theta):
#     """
#     Generate 3D rotation matrix about Y-axis
#     Input:  theta: rotation angle along y-axis
#     Output: Rotation matrix (3 x 3 array)
#     """
#     rot = [[np.cos(theta), 0, np.sin(theta)],[0, 1, 0],[-1*(np.sin(theta)), 0, np.cos(theta)]]
#     return rot


# def part1():
#     # TODO: Solution for Q1
#     # Task 1: Use rotY() to generate cube.gif

#     # Task 2:  Use rotX() and rotY() sequentially to check
#     # the commutative property of Rotation Matrices

#     # Task 3: Combine rotX() and rotY() to render a cube
#     # projection such that end points of diagonal overlap
#     # Hint: Try rendering the cube with multiple configrations
#     # to narrow down the search region

#     # RotationMatrix = [rotY(0),rotY(np.pi/4),rotY(np.pi/2),rotY(3*np.pi/4),rotY(np.pi),rotY(5*np.pi/4),rotY(3*np.pi/2),rotY(7*np.pi/4)]
#     # RotationMatrix = [np.dot(rotY(np.pi/4),rotX(np.pi/4))]
#     RotationMatrix = [np.dot(rotX(np.pi/5),rotY(np.pi/4))]
#     generate_gif(RotationMatrix, file_name='cube.gif')
#     pass


# def split_triptych(trip):
#     """
#     Split a triptych into thirds
#     Input:  trip: a triptych (H x W matrix)
#     Output: R, G, B martices
#     """

#     if( trip.shape[0] % 3  ==  1):
#         trip = np.delete(trip, 0,0)
#     if(trip.shape[0] % 3  == 2):
#         trip = np.delete(trip, 0, 0) #delete first and last rows
#         trip = np.delete(trip, trip.shape[0] - 1,0)

#     R, G, B = np.vsplit(trip,3)
#     # TODO: Split a triptych into thirds and
#     # return three channels as numpy arrays

#     return R, G, B


# def normalized_cross_correlation(ch1, ch2):
#     """
#     Calculates similarity between 2 color channels
#     Input:  ch1: channel 1 matrix
#             ch2: channel 2 matrix
#     Output: normalized cross correlation (scalar)
#     """
#     ch1copy = ch1.copy()
#     ch2copy = ch2.copy()
#     ch1copy = ch1copy.flatten()
#     ch2copy = ch2copy.flatten()
#     NormVal1 = np.linalg.norm(ch1)
#     NormVal2 = np.linalg.norm(ch2)

#     aNorm = ch1 / NormVal1
#     bNorm = ch2 / NormVal2
#     return np.dot(aNorm, bNorm.T)


# def best_offset(ch1, ch2, metric, Xrange=np.arange(-10, 10),
#                 Yrange=np.arange(-10, 10)):
#     """
#     Input:  ch1: channel 1 matrix
#             ch2: channel 2 matrix
#             metric: similarity measure between two channels
#             Xrange: range to search for optimal offset in vertical direction
#             Yrange: range to search for optimal offset in horizontal direction
#     Output: optimal offset for X axis and optimal offset for Y axis

#     Note: Searching in Xrange would mean moving in the vertical
#     axis of the image/matrix, Yrange is the horizontal axis
#     """
#     # TODO: Use metric to align ch2 to ch1 and return optimal offsets
#     MaxMetric = np.linalg.norm(normalized_cross_correlation(ch1,ch2))
#     MaxIdx = 0, 0
#     ch2Copy = ch2.copy()

#     for x in Xrange:
#         for y in Yrange:
#             ch2copy = np.roll(ch2Copy,x,0)
#             ch2copy = np.roll(ch2copy,y,1)

#             score = normalized_cross_correlation(ch1,ch2copy)

#             if(np.linalg.norm(score) > MaxMetric):
#                 MaxIdx = x, y
#                 MaxMetric = np.linalg.norm(score)

#     print(MaxIdx)
#     return MaxIdx


# def align_and_combine(R, G, B, metric):
#     """
#     Input:  R: red channel
#             G: green channel
#             B: blue channel
#             metric: similarity measure between two channels
#     Output: aligned RGB image
#     """
#     # TODO: Use metric to align the three channels

#     # Hint: Use one channel as the anchor to align other two
#     RCopy = R.copy()
#     GCopy = G.copy()
#     BCopy = B.copy()

#     x, y = best_offset(RCopy, GCopy, metric, Xrange=np.arange(-10, 10), Yrange=np.arange(-10, 10))

#     G = np.roll(G,x,0)
#     G = np.roll(G,y,1)



#     x, y = best_offset(RCopy, BCopy, metric, Xrange=np.arange(-10, 10), Yrange=np.arange(-10, 10))

#     B = np.roll(B,y,1)
#     B = np.roll(B,x,0)



#     clr_img = np.stack((B,G,R),2)
#     return clr_img




# def pyramid_align(ref, tar, level):
#     # TODO: Reuse the functions from task 2 to perform the
#     # image pyramid alignment iteratively or recursively
#     # ref == image

#     R, G, B = split_triptych(ref)
#     R.astype(np.int8)
#     G.astype(np.int8)
#     B.astype(np.int8)

#     Rcopy = R.copy()
#     Bcopy = B.copy()
#     Gcopy = G.copy()

#     idx = 0
#     while idx <= 25:
#         Rcopy = np.delete(Rcopy, 0, 0)
#         Rcopy = np.delete(Rcopy, Rcopy.shape[0] - 1, 0)
#         Rcopy = np.delete(Rcopy, 0, 1)
#         Rcopy = np.delete(Rcopy, Rcopy.shape[1] - 1, 1)

#         Bcopy = np.delete(Bcopy, 0, 0)
#         Bcopy = np.delete(Bcopy, Bcopy.shape[0] - 1, 0)
#         Bcopy = np.delete(Bcopy, 0, 1)
#         Bcopy = np.delete(Bcopy, Bcopy.shape[1] - 1, 1)

#         Gcopy = np.delete(Gcopy, 0, 0)
#         Gcopy = np.delete(Gcopy, Gcopy.shape[0] - 1, 0)
#         Gcopy = np.delete(Gcopy, 0, 1)
#         Gcopy = np.delete(Gcopy, Gcopy.shape[1] - 1, 1)
#         idx += 1


#     new1R = cv2.resize(Rcopy, dsize=(int(Rcopy.shape[0] / 4), int(Rcopy.shape[1] / 4)))
#     new1B = cv2.resize(Bcopy, dsize=(int(Rcopy.shape[0] / 4), int(Rcopy.shape[1] / 4)))
#     new1G = cv2.resize(Gcopy, dsize=(int(Rcopy.shape[0] / 4), int(Rcopy.shape[1] / 4)))


#     new2R = cv2.resize(Rcopy, dsize=(int(Rcopy.shape[0] / 16), int(Rcopy.shape[1] / 16)))
#     new2B = cv2.resize(Bcopy, dsize=(int(Rcopy.shape[0] / 16), int(Rcopy.shape[1] / 16)))
#     new2G = cv2.resize(Gcopy, dsize=(int(Rcopy.shape[0] / 16), int(Rcopy.shape[1] / 16)))



#     x2, y2 = best_offset(new2R, new2B, 1, Xrange=np.arange(-10, 10), Yrange=np.arange(-10, 10))

#     new1B = np.roll(new1B, x2*4,0)
#     new1B = np.roll(new1B, y2*4, 1)

#     x2, y2 = best_offset(new2R, new2G, 1, Xrange=np.arange(-10, 10), Yrange=np.arange(-10, 10))

#     new1G = np.roll(new1G, x2*4,0)
#     new1G = np.roll(new1G, y2*4, 1)


#     x1, y1 = best_offset(new1R, new1B, 1, Xrange=np.arange(-10, 10), Yrange=np.arange(-10, 10))

#     Bcopy = np.roll(Bcopy, x1*4,0)
#     Bcopy = np.roll(Bcopy, y1*4,1)


#     x1, y1 = best_offset(new1R, new1G, 1, Xrange=np.arange(-10, 10), Yrange=np.arange(-10, 10))

#     Gcopy = np.roll(Gcopy, x1*4,0)
#     Gcopy = np.roll(Gcopy, y1*4,1)



#     x, y = best_offset(Rcopy, Bcopy, 1, Xrange=np.arange(-10, 10), Yrange=np.arange(-10, 10))

#     Bcopy = np.roll(Bcopy, x,0)
#     Bcopy = np.roll(Bcopy, y,1)


#     x, y = best_offset(Rcopy, Gcopy, 1, Xrange=np.arange(-10, 10), Yrange=np.arange(-10, 10))

#     Gcopy = np.roll(Gcopy, x,0)
#     Gcopy = np.roll(Gcopy, y,1)


#     pyr_img = np.stack((Bcopy,Gcopy,Rcopy),2)


#     plt.imshow(pyr_img)
#     plt.imsave('pyramidImg2.jpg', pyr_img)
#     plt.show()
#     pass


def part1():
    # TODO: Solution for Q2
    # Task 1: Generate a colour image by splitting
    # the triptych image and save it

    # Task 2: Remove misalignment in the colour channels
    # by calculating best offset

    # Task 3: Pyramid alignment
    # idx1 = 1
    # while idx1 < 7:

    #     img_array = plt.imread("prokudin-gorskii/fav"+str(idx1)+".jpg")


    #     R, G, B = split_triptych(img_array)
    #     R.astype(np.int8)
    #     G.astype(np.int8)
    #     B.astype(np.int8)



    #     clr_img = np.stack((B,G,R),2)
    #     plt.imshow(clr_img)
    #     plt.imsave('testOut.jpg', clr_img)
    #     plt.show()


    #     idx = 0
    #     while idx <= 25:
    #         R = np.delete(R, 0, 0)
    #         R = np.delete(R, R.shape[0] - 1, 0)
    #         R = np.delete(R, 0, 1)
    #         R = np.delete(R, R.shape[1] - 1, 1)

    #         B = np.delete(B, 0, 0)
    #         B = np.delete(B, B.shape[0] - 1, 0)
    #         B = np.delete(B, 0, 1)
    #         B = np.delete(B, B.shape[1] - 1, 1)

    #         G = np.delete(G, 0, 0)
    #         G = np.delete(G, G.shape[0] - 1, 0)
    #         G = np.delete(G, 0, 1)
    #         G = np.delete(G, G.shape[1] - 1, 1)
    #         idx += 1


    #     metric = normalized_cross_correlation(R, B)
    #     out = align_and_combine(R,G,B,metric)
    #     plt.imshow(out)
    #     plt.imsave('testAlign'+str(idx1)+'.jpg', out)
    #     plt.show()
    #     idx1 = idx1 + 1


    # pyr_array = plt.imread("tableau/vancouver_tableau.jpg")

    # pyramid_align(pyr_array,0,0)


    # indoorImg = plt.imread("rubik/outdoor.png")
    # #3.1.1


    # image = cv2.cvtColor(indoorImg, cv2.COLOR_BGR2LAB )

    # B = image[:,:,0]
    # G = image[:,:,1]
    # R = image[:,:,2]
    # R.astype(np.int8)
    # G.astype(np.int8)
    # B.astype(np.int8)
    # plt.imshow(B,cmap=plt.get_cmap('gray'))
    # plt.imsave('Bchannel.jpg', B,cmap='gray')
    # plt.imshow(R,cmap=plt.get_cmap('gray'))
    # plt.imsave('Rchannel.jpg', R,cmap='gray')
    # plt.imshow(G,cmap=plt.get_cmap('gray'))
    # plt.imsave('Gchannel.jpg', G,cmap='gray')
    # plt.show()

    #3.3 CROP AND RESIZE
    # image = plt.imread('im2.jpg')

    # # Convert image to array
    # image_arr = np.array(image)

    # # Crop image
    # image_arr = cv2.resize(image_arr, dsize=(256, 256))

    # plt.imsave('im2.jpg', image_arr)
    # plt.show()
    # import required module


    # assign directory
    # directory = 'layers/'

    # iterate over files in
    # that directory
    # folder = Path(directory).glob('*')
    # naming = 0

    # for imgName in folder:
    #     print(imgName)

    #     imgArray = plt.imread(imgName)
    #     print(imgArray.shape)
    #     singleImg = np.array_split(imgArray, 30,1)
    #     idx = 0
    #     while idx < 30:
    #         naming = naming + 1
    #         plt.imsave(str(naming)+'.png', singleImg[idx])
    #         plt.show()
    #         idx = idx + 1
    #         return



    # tar = '840/1.png'
    # src = '1.png'

    # for i in range(1, 841):
    #     os.rename( f"{i}.png", f"{i}/{i}.png")
    # directory = f'Maniak5'
    # idka = 60
    # for filename in os.listdir(directory):
    #     idka = idka + 1
    #     os.rename( filename, str(idka)+".png")


    #for x in range(1,2):

    #metadata location
    # json_file = open("C:/Users/localadmin/Documents/hashlips_art_engine-main/build/json/"+str(x)+".json","r")
    # variables = json.load(json_file)

    # #for debugging
    # # variables["image"] = "ipfs://QmUHQErwbX7bbPrdcYyvmyR2xsKjL8FvLJM1vCYxyee5nk/NFT"+str(x)+".gif"
    # # print(variables["image"])

    # json_file = open("C:/Users/localadmin/Documents/hashlips_art_engine-main/build/json/"+str(x)+".json","w")
    # json.dump(variables,json_file)
    # json_file.close()

    # actualspeed = (.9*(float(actualspeed) * -.99 + 172.67)) #converts metadata speed to a reasonable GIF speed

    # fp_in = f"Maniak2/*.png" # IN: Folder that the images are in
    # fp_out = f"ManiakOut2/NFT1.gif" #OUT: GIF name location

    # #https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif

    # GifDuration = 100 # SET TO WHATEVER YOU WANT (play with this)
    # img, *imgs = [Image.open(f) for f in sorted(glob.glob(fp_in))]
    # img.save(fp=fp_out, format='GIF', append_images=imgs, save_all=True, duration=int(GifDuration), loop=0)

    # img1 = plt.imread("Maniak1/1a (1).png")
    # img2 = plt.imread("Maniak1/1b (1).png")
    # img3 = plt.imread("Maniak1/1c (1).png")
    # img4 = plt.imread("Maniak1/1d (1).png")
    # img5 = plt.imread("Maniak1/1e (1).png")
    # img6 = plt.imread("Maniak1/1f (1).png")
    # # #3.1.1
    # B = img1[:,:,0]
    # G = img1[:,:,1]
    # R = img1[:,:,2]
    # A = img1[:,:,3]
    # clr_img1 = np.stack((B,G,R),2)
    # B2 = img2[:,:,0]
    # G2 = img2[:,:,1]
    # R2 = img2[:,:,2]
    # A2 = img2[:,:,3]
    # clr_img2 = np.stack((B2,G2,R2),2)
    # B3 = img3[:,:,0]
    # G3 = img3[:,:,1]
    # R3 = img3[:,:,2]
    # A3 = img3[:,:,3]
    # clr_img3 = np.stack((B3,G3,R3),2)
    # B4 = img4[:,:,0]
    # G4 = img4[:,:,1]
    # R4 = img4[:,:,2]
    # A4 = img4[:,:,3]
    # clr_img4 = np.stack((B4,G4,R4),2)
    # B5 = img5[:,:,0]
    # G5 = img5[:,:,1]
    # R5 = img5[:,:,2]
    # A5 = img5[:,:,3]
    # clr_img5 = np.stack((B5,G5,R5),2)
    # B6 = img6[:,:,0]
    # G6 = img6[:,:,1]
    # R6 = img6[:,:,2]
    # A6 = img6[:,:,3]
    # clr_img6 = np.stack((B6,G6,R6),2)
    # print(np.amax(clr_img1))
    # # image2 = cv2.cvtColor(img2, cv2.COLOR_BGR2LAB )
    # # image3 = cv2.cvtColor(img3, cv2.COLOR_BGR2LAB )
    # # image4 = cv2.cvtColor(img4, cv2.COLOR_BGR2LAB )
    # plt.imsave('Maniak2/im1.jpg',  clr_img1)
    # plt.show()
    # plt.imsave('Maniak2/im2.jpg', clr_img2)
    # plt.show()
    # plt.imsave('Maniak2/im3.jpg', clr_img3)
    # plt.show()
    # plt.imsave('Maniak2/im4.jpg', clr_img4)
    # plt.show()
    # plt.imsave('Maniak2/im5.jpg', clr_img5)
    # plt.show()
    # plt.imsave('Maniak2/im6.jpg', clr_img6)
    # plt.show()
    # fp_in = f"Maniak2/*.png" # IN: Folder that the images are in
    # fp_out = f"ManiakOut2/NFT1.gif" #OUT: GIF name location

    # #https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif

    # GifDuration = 100 # SET TO WHATEVER YOU WANT (play with this)
    # img, *imgs = [Image.open(f) for f in sorted(glob.glob(fp_in))]
    # img.save(fp=fp_out, format='GIF', append_images=imgs, save_all=True, duration=int(GifDuration), loop=0)



    # looping through all the main directories
    # for folder_num in range(1, 6):
    #     dir = f"build/Maniak{folder_num}"


    #     #      # Now we loop through all the images and copy them to a new name with the letter a
    #     for filename in os.listdir(dir):
    #         #print(dir/filename)
    #         path = dir+"/"+filename
    #         patha = dir+"/d"+filename

    #         #os.system(f"cp {filename}.png a{filename}.png")

    #         shutil.copy(path, patha)
    #         #shutil.copy(path, pathb)
    #         # os.system(f'copy {path} {pathb}')
    #         print(patha)






    for i in range(1, 6):
        image_folder=f'build/Maniak'+str(i)+'/'
        fps= 7.2


        image_files = [os.path.join(image_folder,img)
                    for img in os.listdir(image_folder)
                    if img.endswith(".png")]
        clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
        clip.write_videofile('Maniak0/testManiak.mp4')

        # my_clip = mpe.VideoFileClip('Maniak2/testManiak.mp4')
        # audio_background = mpe.AudioFileClip('meow.mp3')
        # final_audio = mpe.CompositeAudioClip([my_clip.audio, audio_background])
        # final_clip = my_clip.set_audio(final_audio)
        # final_clip.write_videofile('Maniak2/testManiak2.mp4')
        if(i % 10 == 1):
            audio = mpe.AudioFileClip("Alien_Trap.mp3")
        elif(i % 10 == 2 ):
            audio = mpe.AudioFileClip("Darkness.mp3")
        elif(i % 10 == 3):
            audio = mpe.AudioFileClip("Devil_s_Delinquent.mp3")
        elif(i % 10 == 4):
            audio = mpe.AudioFileClip("Emergency_Broadcast.mp3")
        elif(i % 10 == 5):
            audio = mpe.AudioFileClip("Humming_Sphinx.mp3")
        elif(i % 10 == 6 ):
            audio = mpe.AudioFileClip("Lost_In_Space.mp3")
        elif(i % 10 == 7):
            audio = mpe.AudioFileClip("MAPC_Symphony_.mp3")
        elif(i % 10 == 8):
            audio = mpe.AudioFileClip("Predator..mp3")
        elif(i % 10 == 9 ):
            audio = mpe.AudioFileClip("Radioactive_Waste.mp3")
        elif(i % 10 == 0):
            audio = mpe.AudioFileClip("Vintage_MAPC.mp3")


        video1 = mpe.VideoFileClip("Maniak0/testManiak.mp4")
        final = video1.set_audio(audio)


        final.write_videofile("ManiakOut1/"+str(i)+".mp4")
    pass


def part3():
    # TODO: Solution for Q3

    # imageObject = Image.open("diamond bg.gif")

    # print(imageObject.is_animated)

    # print(imageObject.n_frames)



    # # Display individual frames from the loaded animated GIF file

    # for frame in range(0,imageObject.n_frames):

    #     imageObject.seek(frame)

    #     imageObject.show()
    #     imageObject.save(str(frame)+".png")
    pass


def main():
    part1()
    #part2()
    #part3()


if __name__ == "__main__":
    main()
