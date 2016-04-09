import os
import numpy as np
import pandas as pd
from sklearn.datasets import dump_svmlight_file

class FastDT:

    def __init__(self, n_trees = 10, type_forest = "bag", max_depth = 3, rho = 0, minfc = 1) :
        self._compiler_name = "ocamlopt"
        self._source_file_name = "FastDT.ml"
        self._executable_file_name = "./FastDT"
        self._to_delete = ["*.cmi","*.cmx","*.o","*.tmp"]
        self._compilation_string = self._compiler_name + " str.cmxa bigarray.cmxa " + self._source_file_name + " -o " + self._executable_file_name 

        self._temp_dt_file_name = "model.tmp" 
        self._temp_pred_file_name = "predictions.tmp"
        self._train_file_name = "train.tmp"
        self._test_file_name = "test.tmp"

        self._boost_bag = type_forest
        self._n_trees = n_trees
        self._max_depth = max_depth
        self._rho = rho
        self._minfc = minfc
        self._verbose = False
        
        self._compile_fast_dt()

    def _print_call_status(self,command_line,status) :
        print("\"" + command_line + "\"" + " executed with code : " + str(status))

    def _compile_fast_dt(self) :
        res = os.system(self._compilation_string)
        for to_delete in self._to_delete :
            os.system("rm " + to_delete)
        if(self._verbose) :
            self._print_call_status(self._compilation_string,res)

    def _execute_training(self,train_file_name, boost_bag, n_trees, max_depth, rho, minfc) :
        params_string = "-" + boost_bag + " " + str(n_trees) + " -maxd " + str(max_depth) + " -rho " + str(rho) + " -minfc " + str(minfc)
        # FastDT [-boost #|-bag #] [-maxd (10)] [-rho (0)] [-minfc (1)] <data>
        execution_string = self._executable_file_name + " " + train_file_name + " " + params_string + " > " + self._temp_dt_file_name
        res = os.system(execution_string)
        if(self._verbose) :
            self._print_call_status(execution_string,res)

    def _execute_prediction(self,test_file_name) :
        # FastDT -load test-tree test-input
        prediction_string = self._executable_file_name + " -load " + self._temp_dt_file_name + " " + test_file_name + " > " + self._temp_pred_file_name
        res = os.system(prediction_string)
        if(self._verbose) :
            self._print_call_status(prediction_string,res)

    def fit(self,X,y) :
        # write train file to specific format
        dump_svmlight_file(X,y,self._train_file_name,zero_based=True)
        self._execute_training(self._train_file_name, self._boost_bag, self._n_trees, self._max_depth, self._rho, self._minfc)

    def predict_proba(self,X) :
         predicted = self.predict(X)
         return predicted

    def predict(self,X) :
        # write test file to specific format
        dump_svmlight_file(X,np.zeros(X.shape[0]),self._test_file_name,zero_based=True)
        # call _exectute_prediction
        self._execute_prediction(self._test_file_name)
        # import the output of the script and return it
        predictions = pd.read_csv(self._temp_pred_file_name,header = None)
        return predictions
	   
    def get_params(self,deep):
        return {"type":self._boost_bag,"max_depth":self._max_depth,"n_estimators":self._n_trees,"rho":self._rho,"mincf":self._minfc}
