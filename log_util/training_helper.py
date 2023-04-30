import os 
import csv 
from itertools import zip_longest
class TrainingTracker:
	def __init__(self,save_path,filename = 'Training') -> None:
		file_path = os.path.join(save_path,f"{filename}.csv")
		self.file = open(file_path,'w+t')
		self.writer = csv.writer(self.file)
		self.keys = []
		self.sep = ','
	def writekvlist(self,kvs):
		extra_keys = kvs.keys() - self.keys
		if extra_keys:
			self.keys.extend(extra_keys)
			self.file.seek(0)
			lines = self.file.readlines()
			self.file.seek(0)
			for (i, k) in enumerate(self.keys):
				if i > 0:
					self.file.write(',')
				self.file.write(k)
			self.file.write('\n')
			for line in lines[1:]:
				self.file.write(line[:-1])
				self.file.write(self.sep * len(extra_keys))
				self.file.write('\n')
		
		d = []
		for k in self.keys:
			v = kvs.get(k)
			if v is not None:
				d.append(v)
			else:
				d.append([])
		export_data = zip_longest(*d, fillvalue = '')
		self.writer.writerows(export_data)
	def writekvs(self, kvs):
		extra_keys = kvs.keys() - self.keys
		if extra_keys:
			self.keys.extend(extra_keys)
			self.file.seek(0)
			lines = self.file.readlines()
			self.file.seek(0)
			for (i, k) in enumerate(self.keys):
				if i > 0:
					self.file.write(',')
				self.file.write(k)
			self.file.write('\n')
			for line in lines[1:]:
				self.file.write(line[:-1])
				self.file.write(self.sep * len(extra_keys))
				self.file.write('\n')
		for (i, k) in enumerate(self.keys):
			if i > 0:
				self.file.write(',')
			v = kvs.get(k)
			if v is not None:
				self.file.write(str(v))
		self.file.write('\n')
		self.file.flush()
	def close(self):
		self.file.close()