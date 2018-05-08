import Redcarpet_assignment

#this is a test function for the add_target() function in the redcarpet_assignment.py which will check for the target column added or not in the date frame
def test_add_target():
	 df = Redcarpet_assignment.add_target()
	 assert df.shape[1] == 12

#This test fun check for the datetype of the column it should not be object after encoding the data while we put into in the algorithm
def test_econding_cat_feature():
	df = Redcarpet_assignment.econding_cat_feature()
	funding_cols = ['Agency', 'Subagency', 'YE', 'FY2008', 'FY2009', 'FY2010', 'MGSTEM','F1) Primary Investment Objective','J) Focus on Underrepresented Groups in STEM','K) Eligibility Restrictions','Q) Legislation Required to Shift Focus?']
	for column in funding_cols:
		assert df[column].dtypes != object
