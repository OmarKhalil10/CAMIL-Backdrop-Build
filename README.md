# CA-MIL Centralized-and-Automated-Medical-Image-Analysis-Laboratory

## Development Setup
1. **Download the project starter code locally**
```
git clone https://github.com/OmarKhalil10/Webstack-Portfolio-Project.git
```

2. **Before and After editing your code, Use the commands below:**

before editing anything pull new changes from GitHub.
```
git pull
```
Once you are done editing, you can push the local repository to your Github account using the following commands.
```
git add .
git commit -m "your comment message"
git push
```

3. **Initialize and activate a virtualenv using:**
```
python -m virtualenv venv
source venv/bin/activate
```
>**Note** - In Windows, the `venv` does not have a `bin` directory. Therefore, you'd use the analogous command shown below:
```
source venv/Scripts/activate
deactivate
```

4. **Install the dependencies:**
```
pip install -r requirements.txt
```

5. **Run the development server:**
```
export FLASK_APP=app.py
export FLASK_ENV=development
export FLASK_DEBUG=true
flask run --reload
```

6. **Verify on the Browser**<br>
Navigate to project homepage [http://127.0.0.1:5000/](http://127.0.0.1:5000/) or [http://localhost:5000](http://localhost:5000)


# Adding Routes
## To add a new page
* Create the html, css, js in the specified folder using the same folder structure.
* Create a new route in the [app.py](./app.py) file with the name you want using only dashes to seperate words.
```PYTHON
@app.route('NEW-ROUTE')
```
* Define your serving function using a unique name not used before in the whole application.
```PYTHON
def NEW_UNIQUE_NAME():
```
* Return your html file path using render_template.
```PYTHON
return render_template('FOLDER_PATH/FILE_PATH.html')
```
* Your newely created route should look like this.
```PYTHON
@app.route('NEW-ROUTE')
def NEW_UNIQUE_NAME():
    return render_template('FOLDER_PATH/FILE_PATH.html')
```

## To run the development server
* Open git bash terminal
```bash
FLASK_APP=app.py
FLASK_ENV=development
flask run --reload
```

# Push to Production
## 1. Set up GCP account, project, etc.

```
export MY_EMAIL_ADDRESS=omar@bamboogeeks.com
export MY_PROJECT_ID=ml-tf-398511
```

## 2. Configure your project.

```
gcloud config set account $MY_EMAIL_ADDRESS
gcloud auth login $MY_EMAIL_ADDRESS
gcloud config set project $MY_PROJECT_ID
```

## 3. Build a Docker image of the Flask application

1. Test docker connectivity
```
docker ps
```

2. Build docker image locally
```
./run_docker.sh
```

## 4. Push the Docker image to Container Registry 

```
docker push gcr.io/$MY_PROJECT_ID/camil:v1
```

### Container Registry Repository

![Container Registry Repository](/static/documentation_files/repository.png)

### Docker image

![Docker image](/static/documentation_files/docker_image.png)

### Docker image details

![Docker image details](/static/documentation_files/docker_image_datails.png)
## 5. Deploy a Docker image on Cloud Run

```
gcloud run deploy camil \
 --image gcr.io/$MY_PROJECT_ID/camil:v1 \
 --region europe-west6 \
 --platform managed \
 --memory 8Gi \
 --cpu 2 \
 --max-instances 25
```

### Cloud Run logs

![Cloud Run](/static/documentation_files/cloud_run_logs.png)


#### Use the gcloud run revisions list command to list all revisions of your Cloud Run service. Replace <SERVICE_NAME> with the name of your service.

```
gcloud run revisions list --platform managed --region europe-west6 --service camil --format="value(name)" | sort
```

#### Output

```
camil-00001-lec
```

#### Use the gcloud run revisions delete command to delete each of the old revisions. Replace <REVISION_NAME> with the name of each revision you want to delete. [If Any!]

```
gcloud run revisions delete <REVISION_NAME> --platform managed --region europe-west6 --quiet
```

#### NOTE
You can run this command for each old revision you copied in the previos step

#### Show the description of a specific revision in Google Cloud Run

```
gcloud run revisions describe camil-00001-lec \
  --platform managed \
  --region europe-west6 \
```

#### Output

```
+ Revision camil-00001-lec in region europe-west6
 
Image:               gcr.io/ml-tf-398511/camil@sha256:ad752b73609fc1f872051485b172ec4358ba1a3bec6db03076b14a49aecae18c
Port:                8080
Memory:              8Gi
CPU:                 2
Service account:     220929236898-compute@developer.gserviceaccount.com
Concurrency:         80
Max Instances:       25
Timeout:             300s
Startup Probe:
  TCP every 240s
  Port:              8080
  Initial delay:     0s
  Timeout:           240s
  Failure threshold: 1
  Type:              Default
```

## 14. Check the Flask application on Cloud Run

```
https://camil-ltnijdawbq-oa.a.run.app
```

Or you send HTTPS requests to the Cloud Run instance for testing

```
python request_main_v6.py
```

#### Output

1. index.html content
2. <Response [200]>


## Problems solved
- [x] The problem we faced is choosing the right disease to work on as diseases vary in the methods that we use to diagnose them, and the data available to use in model training
- [x] The Second problem is collecting data we need the data to have some important features and it also must be recent because we need to train our model on the most recent chest radiography images to get more accurate prediction mutation rate
- [x] After that we needed large number of images to be able to cover almost every scenario to achieve better prediction accuracy
- [x] After that we faced is how we could train this model with this amount of data and we solve this problem by using Azure platform which provides us with higher compute sources to build and train our model.
- [x] Finally, that we faced another problem as we wanted to use our trained model to send chest radiography images from our website and receive the result so we had to learn how to use the model.h file and the model history in our website backend to send and receive requests between the website and the model.
- [x] convert .h5 models to .onnx to be able to push to production on GCP production environment.
 
## Future Work
- [ ] validate emails sent using both news letter and contact us
- [ ] make muiltiple version of the logo with multiple sizes to fix different screen sizes
- [ ] create a database with all required data about vaccination centers, doctors, pharmacies and hospitals
- [ ] add a 2 tables to this database, one for contact us and another one for our news letter subscipers
- [ ] add more detectors for diseases like ones mentioned in the sides for future work
- [ ] store your users related files to Google Cloud Storage bucket and interact with it using APIs
- [ ] create admin page that can be accessed via the website and used to edit website content
- [ ] choose an Email Marketing Platform to interact with subscribers with latest website related updates
- [ ] work on improving our model algorithm to make better predictions and help in prevention of diseases. 

## Contributions

Contributions and enhancements to **CA-MIL** are welcome! Feel free to fork the repository, make improvements, and submit pull requests.

## License

This project is licensed under the MIT License. See the [LICENSE](https://github.com/OmarKhalil10/Webstack-Portfolio-Project/blob/main/LICENSE) file for details.

## Authors

- [Omar Khalil](https://github.com/OmarKhalil10)

## Contact

If you have any questions or suggestions, please feel free to [contact me](mailto:omar.khalil498@gmail.com)