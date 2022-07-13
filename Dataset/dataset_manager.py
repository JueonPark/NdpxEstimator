import openpyxl
import pandas as pd

class DatasetManager:
	def fetch_data():
		# Data to fetch from the workbook:
		# [x]:
		# - ShapeSize
		# - #input
		# - #output
		# - #op
		# y:
		# - RealNdpxCost
		workbook = openpyxl.load_workbook("ndpx_estimation_data.xlsx")
		dataset_sheet = workbook["dataset"]
		dataset = pd.DataFrame(dataset_sheet.values)
		return dataset

if __name__ == "__main__":
	print(DatasetManager.fetch_data())