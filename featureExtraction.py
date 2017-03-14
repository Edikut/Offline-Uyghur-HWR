import numpy as np
import cv2		# import OpenCV module
from os import listdir
from scipy.misc import imread


class Compile:

	# for 96*96 image
	M = 96
	N = 96
	############################################
	# for MNIST datasets
	root_path_MNIST = "E:/HWRProj/Data/MNISTImage/"
	img_data = "t10k-images.idx3-ubyte"
	img_label = "t10k-labels.idx1-ubyte"
	DATA_HEAD = 16		# bytes
	LABEL_HEAD = 8		# bytes
	IMAGE_WIDTH = 28
	IMAGE_HEIGHT = 28
	IMAGE_SIZE = IMAGE_WIDTH*IMAGE_HEIGHT
	LABEL_SIZE = 1		# byte
	############################################
	root_path = "E:/HWRProj/Data/GeneratedImage/"
	# feat_path = "E:/HWRProj/Data/FeaturesData/mnist_wt1414.csv"		# for 96*96 image
	feat_path = "E:/HWRProj/Data/FeaturesData/smlImg/fullImg.csv"

	feat_obj = ""
	ft_handle = ""

	def __init__(self):
		self.feat_obj = Moment()
		ft_handle = self.feat_obj.moment_feature
		print('Compile class instanciated!')

	def _open_file(self, m_file, wr = "r"):
		try:
			file = open(m_file,wr)
			return file
		except:
			print("error! opennnig file: %s\n" %m_file)
			return False

	def _image_read(self, m_file):
		try:
			file = imread(m_file, "r")
			return file
		except:
			print("error! opening file: %s\n" %m_file)
			return False

	def _size_normalize(self, img):	# to fill image's size to M X N
		standard_size = self.M * self.N
		if img.size < standard_size:
			norm_img = np.zeros([self.M, self.N])
			norm_img[0:img.shape[0], 0:img.shape[1]] = img
			return norm_img
		else: return img

	def _print_ftr_to_csv_format(self, feat_file, letter_label, ft_list):
		"""
		write features with csv (comma separated values) file format
		<value1>,<value2>,...,<label>
		:param feat_file:		feature file
		:param letter_label:	<label> for features
		:param ft_list:			<valuek> ...(k = 1,2,...)
		:return:				file handle 'feat_file'
		"""
		ft_line = ""
		for h in ft_list:
			ft_line += str(h) + ","
		feat_file.write(ft_line + str(letter_label) + "\n")

	def _print_ftr_to_svm_format(self, feat_file, letter_label, ft_list):
		"""
		write features with libsvm format
		<label> <index1>:<value1> <index2>:<value2> ...
		:param feat_file:		feature file
		:param letter_label:	<label> for features
		:param ft_list:			<valuek> ...(k = 1,2,...)
		:return:				file handle 'feat_file'
		"""
		feat_index = 1  # dimension index
		ft_line = letter_label
		for h in ft_list:
			ft_line = ft_line + " " + str(feat_index) + ":" + str(h)
			feat_index += 1
		feat_file.write(ft_line + "\n")

	def feature_extract_by_person(self, root_path, start, end, ft_func_handle=None, csv=False):
		"""
		feature extracts from letter images in the writers folder from the 'start' to 'end'
		:param root_path:
		:param start: 	start folder
		:param end: 	end folder
		:param ft_func_handle: feature extraction method
		:param csv: switch for svm or csv file format, default format is libsvm file
		:return:
		"""
		feat_file = self._open_file(self.feat_path, wr = "w")
		for path_list in self.path_each_person(root_path, start, end):
			# path_list is 128 letters' full path wrote by someone (one of 'end - start' people)
			for path_file in path_list:
				# path_file is one letters' (one of 128 letters) full path wrote by someone
				# /////////////path trimming for letter label
				letter_label = path_file[:-4]  # to trim extension '.bmp'
				indx = letter_label.rfind("/") + 1  # positioning image name's start position
				letter_label = letter_label[indx:]  # get image name
				letter_label = letter_label.replace('_', '')  # to trim '_' from label
				# ////////////path trimming

				img = self._image_read(path_file)
				# img = (255 - img)		# exchange background & foreground color
				if not path_file:
					continue

				# special for full image as a feature array
				# no need to extract any feature
				if type(img) == np.ndarray:
					ft_list = img.flatten()
					ft_list = ft_list.tolist()

				# norm_img = self._size_normalize(img)			# if image size smaller than M*N
				# ft_list = ft_func_handle(norm_img)				# features extracted by ft_func_handle

				# following line for blockproc method
				# ft_list = ft_func_handle(img, fun=self.feat_obj.weight, blk_size=[28, 1])
				# if type(ft_list) == np.ndarray:		# to check ft_list is 'list' or not
				# 	ft_list = ft_list.flatten()
				# 	ft_list = ft_list.tolist()

				if len(ft_list):
					if csv:		# if we need csv (.csv format) is true
						self._print_ftr_to_csv_format(feat_file, letter_label, ft_list)
					else:
						self._print_ftr_to_svm_format(feat_file, letter_label, ft_list)
		feat_file.close()

	def letter_sample_extract(self, root_path, start, end):
		for path_list in self.path_each_letter(root_path, start, end):
			summation = [] # np.zeros([self.M, self.N])
			for hfile in path_list:
				img = self.image_read(hfile)
				if not hfile:
					continue
				norm_img = self.size_normalize(img)
				summation.append(norm_img) # += norm_img
			yield summation

	def path_each_person(self, root_path, start_idx, end_idx):
		dirs = listdir(root_path)
		for x in range(start_idx, end_idx):
			person_path = root_path + dirs[x]
			ltr_list = listdir(person_path)
			ltr_full_path = []
			for y in ltr_list:
				ltr_full_path.append(person_path + '/' + y)
			yield ltr_full_path
			print("{}st person over".format(x + 1))

	def path_each_letter(self, root_path, start_idx, end_idx):
		dirs = listdir(root_path)
		person_path = root_path + dirs[0]
		if len(person_path) != 128:
			print('error2! letter count is not 128!!!, ignored')
		ltrs = listdir(person_path)
		print('Feature Extraction Started!')
		for ltr in ltrs:
			ltr_full_path = []
			for dir in dirs:     # extract features from first 300 person's sample for train, and last 96 for test
				ltr_full_path.append(root_path + dir + '/' + ltr)
			yield ltr_full_path

	def _load_mnist_image(self, f_img, f_lbl):
		# img = f_img.read(self.IMAGE_SIZE)
		img = np.fromfile(f_img, dtype=np.uint8, sep="", count=self.IMAGE_SIZE)
		if type(img) != np.ndarray:
			print("Error: Reading image occur a problem!")
			pass
		img = img.reshape(self.IMAGE_WIDTH, self.IMAGE_HEIGHT)
		# lbl = f_lbl.read(self.LABEL_SIZE)
		lbl = np.fromfile(f_lbl, dtype=np.uint8, sep="", count=self.LABEL_SIZE)
		if type(lbl) != np.ndarray:
			print("Error: Reading label occur a problem!")
			pass
		lbl = lbl[0]
		return img, lbl

	def feature_extract_from_mnist_digits(self, start, end, ft_func_handle, csv=False):
		feat_file = self._open_file(self.feat_path, wr="w")
		f_img = self._open_file(self.root_path_MNIST + self.img_data, "rb")
		f_img.seek(self.DATA_HEAD, 0)		# flip file head, 0 which means absolute file positioning
		f_lbl = self._open_file(self.root_path_MNIST + self.img_label, "rb")
		f_lbl.seek(self.LABEL_HEAD, 0)		# flip file head
		flg_start = True
		for pos in range(start, end):
			if flg_start:		# if first loop, move the position indicator
				f_img.seek(self.IMAGE_SIZE*(start - 1), 1)		# locate the position start to read
				f_lbl.seek(self.LABEL_SIZE*(start - 1), 1)		# locate the position start to read
				flg_start = False
			img, letter_label = self._load_mnist_image(f_img, f_lbl)

			########################################
			# import matplotlib.pyplot as plt
			# plt.imshow(img)
			# plt.show()

			# ft_list = ft_func_handle(img)  # features extracted by ft_func_handle
			# following line for blockproc method
			ft_list = ft_func_handle(img, fun=self.feat_obj.weight, blk_size=[14, 14])
			if type(ft_list) == np.ndarray:  # to check ft_list is 'list' or not
				ft_list = ft_list.flatten()
				ft_list = ft_list.tolist()

			if len(ft_list):
				if csv:  # if we need csv (.csv format) is true
					self._print_ftr_to_csv_format(feat_file, letter_label, ft_list)
				else:
					self._print_ftr_to_svm_format(feat_file, letter_label, ft_list)
			print("{} st image processed!".format(pos))
		f_img.close()
		f_lbl.close()
		feat_file.close()


# //////////////////////////////////////////////////////////////
# //////////////////////////////////////////////////////////////
class Features:

	def blockproc(self,  M, fun = None, blk_size=[8, 8], overlap=(0, 0)):
		if M.shape[0] < blk_size[0] or M.shape[1] < blk_size[1]:
			print("error, block size is larger than Matrix size!")
			return None
		# padding zeros to M and copy to A
		pad = [blk_size[0] - M.shape[0] % blk_size[0], blk_size[1] - M.shape[1] % blk_size[1]]
		if pad[0] >= blk_size[0]:
			pad[0] = 0
		if pad[1] >= blk_size[1]:
			pad[1] = 0
		A = np.zeros([M.shape[0] + pad[0], M.shape[1] + pad[1]])
		A[:M.shape[0], :M.shape[1]] = M
		S = np.zeros(A.shape)  # assume the returning array is no larger than M
		I = list(range(0, A.shape[0] + 1, blk_size[0]))
		J = list(range(0, A.shape[1] + 1, blk_size[1]))
		# rr=np.zeros([1,1])
		top = [0, 0]
		hv = [len(I), len(J)]
		for i in range(1, hv[0]):
			for j in range(1, hv[1]):
				block = A[I[i - 1]:I[i], J[j - 1]:J[j]]
				#  //////////////////////
				rr = fun(block)
				# /////////////////////
				end = [top[0] + rr.shape[0], top[1] + rr.shape[1]]
				S[top[0]:end[0], top[1]:end[1]] = rr
				if j >= hv[1] - 1:
					top[0] = end[0]
					top[1] = 0
				else:
					top[1] = end[1]
		return S[:end[0], :end[1]]  # np.concatenate(rows, axis=0)

	def _feature_export(self, arr_list, m_list):
		"""
		merge two lists
		:param arr_list: list to be merged to the end of 'm_list'
		:param m_list:   merged list
		:return: 		 m_list
		"""
		if not str(type(arr_list)) == "<class 'list'>":
			arr_list = arr_list.flatten()
		for i in arr_list:
			m_list.append(i)


	def weight(self, block):
		S = np.zeros([1, 1])  # weight of each block is a single integer
		for i in range(0, block.shape[0]):
			for j in range(0, block.shape[1]):
				if block[i, j] >= 100:		# for MNIST images
					S += 1
		return S


class Moment(Features):

	def moment_feature(self, norm_img):
		"""
		get 'moment' and 'Hu moment' invariants as a kind of feature
		:param norm_img: size normalized image array
		:return:	feature list 'm_list'
		"""
		# m_list = []
		mmt = cv2.moments(norm_img)     		# get moment invariants, mmt is dictionary
		hmmt = cv2.HuMoments(mmt)       		# get Hu moments, hmmt is numpy array
		# hmmt2 = -np.sign(hmmt) * np.log10(np.abs(hmmt))
		hmmt2arr = hmmt.flatten()           	# 2-d array to 1-d array
		hm_list = hmmt2arr.tolist()         	# array to list
		# m_list = [v for v in mmt.values()]		# dictionary to list for moments
		# self._feature_export(hm_list, m_list)	# merge two kind of moments in a list, 'm_list'
		return hm_list


# ///////////////////////////////////////////////////
# ///////////////////////////////////////////////////

if __name__ == "__main__":

	print('Feature Extraction Started!')
	mm = Compile()
	# mm.feature_extract_from_mnist_digits(1, 1001, mm.feat_obj.moment_feature, csv=True)
	# mm.feature_extract_from_mnist_digits(1, 1001, mm.feat_obj.blockproc, csv=True)
	# mm.feature_extract_by_person(mm.root_path, 0, 396, mm.feat_obj.moment_feature, csv=True)
	# mm.feature_extract_by_person(mm.root_path, 0, 396, mm.feat_obj.blockproc, csv=True)

	# becoz full image as a feature array, no need to special feature extraction method
	# so, not used any FE method
	mm.feature_extract_by_person(mm.root_path, 0, 396, csv=True)

	print('Mission Complete!')



