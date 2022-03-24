Project Files:

-> COMP_6721_CNN.py: File for CNN model<br/>
-> Config.py: File for all the configuration of the Application<br/>
-> Trainer.py: File that contains code for model training<br/>
-> Predict_samples.py: Model predictions on 100 sample images<br/>
-> Predict_image.py: To make prediction on a single image.<br/>
-> Expectation of originality form.<br/>
-> Report_COMP 6721_PHASE_1_NS_05_1.pdf: Report for the project.<br/>

Training Data: https://drive.google.com/drive/folders/1a7Ixc-HeBQqdzhViJl4lFMs4Hnd5PL1k  <br/>



How to start training:<br/>
-> Config.py contains constant related to configs ,where the necessary changes can be made such as data location, where to save model,name of the save model etc.<br/>
	DIR_PATH ==> path to the Training Data set (link for sample images is given above)
	PREDICTION_DIR_PATH ==> path to the Training Data set (link for sample images is given above)
	SAVE_MODEL_NAME ==> name of the model that is being saved after training
-> After all the changes are made run: python Trainer.py<br/>
-> After the training is completed evaluation will be shown, loss and acccuracy graph would be plotted, also the classification report and confusion matrix would be displayed.<br/>

To run the model on sample of 100 images:<br/>
python Predict_samples <br/>

To run the model on sample image:<br/>
python Predict_image <br/>

