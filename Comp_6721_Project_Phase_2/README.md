Project Files:

-> COMP_6721_CNN.py: File for CNN model<br/>
-> Config.py: File for all the configuration of the Application<br/>
-> Trainer.py: File that contains code for model training<br/>
-> Predict_samples.py: Model predictions on 100 sample images<br/>
-> Predict_image.py: To make prediction on a single image.<br/>
-> bias_detector.py: To detect bias.<br/>
-> Expectation of originality form.<br/>
-> Report_COMP 6721_PHASE_2_NS_05_1.pdf: Report for the project.<br/>
-> k-foldModel: New trained k-fold model.<br/>
-> oldmodel: Old model.<br/>

Training Data: 
Dataset : https://drive.google.com/drive/folders/1GFDYJ6HRef5dLp-jKOR1ppKasZR4clEB?usp=sharing 
Bias Seperated Data : https://drive.google.com/drive/folders/1GCrbVRcL_cuHJYvlXyEDra5eB1KzpkV1?usp=sharing


How to start training:<br/>
-> Config.py contains constant related to configs ,where the necessary changes can be made such as data location, where to save model,name of the save model etc.<br/>
	DIR_PATH ==> path to the Training Data set (link for sample images is given above)
	PREDICTION_DIR_PATH ==> path to the Training Data set (link for sample images is given above)
	SAVE_MODEL_NAME ==> name of the model that is being saved after training
	GENDER_PATH ==> path to gender seperated data(Available in Bias seperated data link mentioned above)
	AGE_PATH ==> path to Age seperated Data(Available in Bias seperated data link mentioned above)
-> After all the changes are made run: python Trainer.py<br/>
-> After the training is completed evaluation will be shown, loss and acccuracy graph would be plotted, also the classification report and confusion matrix would be displayed.<br/>

To run the model on sample of 100 images:<br/>
python Predict_samples <br/>

To run the model on sample image:<br/>
python Predict_image <br/>

To run the bias:<br/>
python bias_detector<br/>


