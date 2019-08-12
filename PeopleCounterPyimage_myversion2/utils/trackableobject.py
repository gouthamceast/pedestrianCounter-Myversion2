class TrackableObject:
	def __init__(self, objectID, centroid):
		self.objectID = objectID
		self.centroids = [centroid]

		# initialize a boolean used to indicate if the object has already been counted or not
		self.counted = False