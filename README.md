# CourseProject

Please fork this repository and paste the github link of your fork on Microsoft CMT. Detailed instructions are on Coursera under Week 1: Course Project Overview/Week 9 Activities.

**Data Set Creation: A safe way to create a privatized dataset that does not leak personal/senitive information**

## Dataset creation

### Sensitive data

We first create a senstivie dataset
![image](https://user-images.githubusercontent.com/96288942/205836247-55bd6a03-44ca-436a-a030-89c6b28a5446.png)

### Create synthetic data using our DPGAN model
Train the DPGAN model
```
# main() function in DPGAN.py
data = load_data()
generator_model = train(data)
```
Use the trained model to generate synthetic data
```
generator = load_model('trained_generator')
synthetic_data = generator(tf.random.normal(data.shape))
```
![image](https://user-images.githubusercontent.com/96288942/205837211-0d89834d-e7cb-4093-85bb-2f67e2145315.png)

### Validate that our synthetic data is similar to the sensitive dataset
![image](https://user-images.githubusercontent.com/96288942/205837447-73e1ee71-e360-495f-a569-f0b999ae3dc0.png)
![image](https://user-images.githubusercontent.com/96288942/205837511-d132abaa-3e85-4e6e-a4b0-2257ff20ac38.png)
![image](https://user-images.githubusercontent.com/96288942/205837580-31435d78-ab98-49ef-a050-e5f555324a09.png)
![image](https://user-images.githubusercontent.com/96288942/205837602-7f55bf82-f427-46e4-a955-a1cebd44b465.png)

### Validate that we cannot reconstruct the *name* of the individual based off the synthetic data
```
df_synthetic = pd.read_csv('data/synthetic_data.csv')
```
Let's try to find Cameron Ward
```
df_synthetic.loc[(df_synthetic['sensitive_feature1'] > -150)&
                 (df_synthetic['sensitive_feature1'] < -100)&
                 (df_synthetic['sensitive_feature2'] > 0)&
                 (df_synthetic['sensitive_feature2'] < 5)&
                 (df_synthetic['sensitive_feature3'] > 0)&
                 (df_synthetic['sensitive_feature3'] < 5)
                ]
```
Which returns no data

# Release dataset to the public!
The final dataset can be found: data/synthetic_data.csv
