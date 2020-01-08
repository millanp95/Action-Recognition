# Action-Recognition
Python implementation for the extraction of Hand-Crafted features to perform action recognition in videos. The models are trained on the UCF-101 database and on a small database called PUJ created specifically for the project. This new database consist on 3 similar classes: Handshake, dropping an object and picking an object from the ground, also one of the objectives was giving the model the ability to tell if none of the actions were happening. This study was created to test the performance of Deep Learning architectures on small databases such as the ones we would get in a surveillance context. 

See ```demo0.mp4 ``` for a demo of the model over the UCF-101 database and ```demo1.mp4 ``` for a demo of the model over the PUJ database. 

## User Guide. 

1. Extracting Trayectories

	Run the following command: ```python iDT.py <path to video>  <Optical Flow Technique (DIS, Deep)>```
	
	Example: ```python iDT.py Videos/new_video.mp4 DIS```

2. Extracting HoG and Hof Features.

	Run the following command:```python ExtracFeatures.py <path to video>  <Optical Flow Technique (DIS, Deep)>```
	
	Example: ```python extractFeatures.py Videos/new_video.mp4 DIS```
 
3. Video classification.

	For PUJ database: 
		Run the following command: ```python ExtracFeatures.py <N> <MachineNumber>  <path to video>  <Optical Flow Technique (DIS, Deep)>```

	For UCF-101 database: 
		Run the following command: ```python ExtracFeatures.py <MachineNumber_N> <MachineNumber>  <path to video>  <Optical Flow Technique (DIS, Deep)>```

```
** Machine Numbers:
	PUJ1---1 (N=20,25,30,35,40,50)
	PUJ2---2 (N=25 Only)
	UCF1---3 (N=30 Only)
	UCF2---4 (N=30 Only)
	UCF3---5 (N=30 Only)
	UCF4---6 (N=50 Only)
````

Example: 
```python Classification.py 40 1 ./Videos/Example1.mp4 DIS```
```python Classification.py 4_30 1 ./Videos/Example3.mp4 Deep```

4. Training with a new Database.

	4.1. Place all your labeled videos in the "Labeled" folder, with the format: class_numer.mp4.
	
	4.2. Make your partition of the data in the files:
	
		Train.txt ---> Train files 
		Test.txt  ---> test Files 
	
		
	4.3. Run the following command: ```python Train.py <number of gaussians>```

5. Testing New Data Set. 

	5.1. Cross Validation, Run the folliwing command.
 		 Run the following command: ```python CrossValidation.py <number of gaussians>```

	5.2 Particular Testing Partition.
		Run the following command: ```python MachineTest.py <number of gaussians> <MachineNumber>```
