TRAIN_TEST_SPLITS = [ 4966, 58023, 19639, 63847, 30619, 58170, 19854, 37524, 37938, 9618]
NUM_FOLDS = 3

train_path='../../data/processed/train'
test_path='../../data/processed/test'

output = '../../data/output/'
b_train_file_name, b_test_file_name = train_path+'/ui_train.csv', test_path+'/ui_test.csv'
b_train_file_name2, b_test_file_name2 = train_path+'/ui_train2.csv', test_path+'/ui_test2.csv'
ci_train_file_name, ci_test_file_name = train_path+'/u_train.csv', test_path+'/u_test.csv'
cii_train_file_name, cii_test_file_name = train_path+'/ub_train.csv', test_path+'/ub_test.csv'
ciii_train_file_name, ciii_test_file_name = train_path+'/uc_train.csv', test_path+'/uc_test.csv'
civ_train_file_name, civ_test_file_name = train_path+'/ua_train.csv', test_path+'/ua_test.csv'




pkl_train_path='../../data/interim/train'
pkl_test_path='../../data/interim/test'
