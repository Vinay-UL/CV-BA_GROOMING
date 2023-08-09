import os
import csv
from PIL import Image
import skimage.io as io
from skimage.color import rgb2gray
from brisque import BRISQUE
import cv2
import brisque
from sklearn.metrics import accuracy_score, precision_score, recall_score
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
import json
from azure.storage.blob import BlobServiceClient, BlobType
import tempfile
import io
import sys
import shutil
from configs.config_business_rules import *
from azure.core.exceptions import ResourceExistsError
from logger.logging_code import CustomLogger

warnings.filterwarnings("ignore")

output_folder = "Outputs"

# Create the folders if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

COMMON_DATE_FORMAT = "%Y-%m-%d"

class CV_MetricsCalculator:

    def __init__(self): 
        
        self.connection_string = connection_string
        self.FL_container_name = FL_container_name
        self.HL_container_name = HL_container_name
        self.actual_predicted_values = actual_predicted_values
        self.upload_files_to_container= upload_files_to_container
        self.blob_service_client = BlobServiceClient.from_connection_string(self.connection_string)
        
        self.output_folder = output_folder
        
        self.Visual_Quality_Outputs = os.path.join(self.output_folder, "Visual_Quality")
        
        os.makedirs(self.Visual_Quality_Outputs, exist_ok=True)

        self.csv_file_FL = open(os.path.join(self.Visual_Quality_Outputs, "Visual_quality_FL.csv"), "w", newline='')
        self.csv_writer_FL = csv.writer(self.csv_file_FL)
        self.csv_writer_FL.writerow(["Date", "Image Name", "Resolution", "Size (bytes)", "Image Path", "Brisque_Score", "Quality_Assessment"])

        self.csv_file_HL = open(os.path.join(self.Visual_Quality_Outputs, "Visual_quality_HL.csv"), "w", newline='')
        self.csv_writer_HL = csv.writer(self.csv_file_HL)
        self.csv_writer_HL.writerow(["Date", "Image Name", "Resolution", "Size (bytes)", "Image Path", "Brisque_Score", "Quality_Assessment"])
        
        self.threshold = 30.0
        
        self.container_client_FL = self.blob_service_client.get_container_client(self.FL_container_name)
        self.container_client_HL = self.blob_service_client.get_container_client(self.HL_container_name)
        
        self.logger = CustomLogger.get_instance(file_name=os.path.join(self.output_folder, LOG_FILE_NAME))
        self.y_true = []
        self.y_pred = []

######### READING FILES FROM CONTAINER AND PREPARATION OF Y_True and Y_Pred labels##############
    def read_files_from_container(self):
        try:
            container_client_preprocess = self.blob_service_client.get_container_client(self.actual_predicted_values)
            actual_folder = os.path.join(self.output_folder, "Actual")
            predicted_folder = os.path.join(self.output_folder, "Predicted")

            if os.path.exists(actual_folder):
                for file_name in os.listdir(actual_folder):
                    file_path = os.path.join(actual_folder, file_name)
                    os.remove(file_path)
                os.rmdir(actual_folder)
            if os.path.exists(predicted_folder):
                for file_name in os.listdir(predicted_folder):
                    file_path = os.path.join(predicted_folder, file_name)
                    os.remove(file_path)
                os.rmdir(predicted_folder)

            # Create the "Actual" and "Predicted" folders
            os.makedirs(actual_folder, exist_ok=True)
            os.makedirs(predicted_folder, exist_ok=True)

            df_actual_list = []  # List to store individual actual dataframes
            df_predicted_list = []  # List to store individual predicted dataframes

            uniform_actual = []
            shoes_actual = []
            blusher_actual = []
            eyeliner_actual = []
            eyeshadow_actual = []
            lipstick_actual = []
            shaved_beard_actual = []
            combed_hairs_actual = []

            uniform_pred = []
            shoes_pred = []
            blusher_pred = []
            eyeliner_pred = []
            eyeshadow_pred = []
            lipstick_pred = []
            shaved_beard_pred = []
            combed_hairs_pred = []

            list_file_name = []
            for blob in container_client_preprocess.list_blobs():
                if "actual" in blob.name.lower():
                    file_name = blob.name.split('/')[-1]
                    list_file_name.append(file_name)
                    dates_dynamic = []
                    # Split the file_name directly
                    date_str = file_name.split('.')[0].split('_')[-1]
                    date = datetime.strptime(date_str, '%d-%m-%Y').date()
                    dates_dynamic.append(date)
                    latest_date_dynamic = max(dates_dynamic)
            file_dynamic = {}
            for file in list_file_name:
                if latest_date_dynamic.strftime("%d-%m-%Y") in file.lower():
                    folder_name = "Actual"
                    if folder_name not in file_dynamic:
                        file_dynamic[folder_name] = [file]
                    else:
                        file_dynamic[folder_name].append(file)
            self.logger.log(message=f"file_dynamic:{str(file_dynamic)}")

             # Read the content using pandas
            actual_list = file_dynamic['Actual']
            for file in actual_list:
                if "half" in file:
                    if file.endswith(f".xlsx"):
                        # Download the blob content
                        blob_client_hl = container_client_preprocess.get_blob_client("Actual/"+file)
                        content_ac_excel_hl = blob_client_hl.download_blob().readall()
                        df_actual_hl = pd.read_excel(io.BytesIO(content_ac_excel_hl))
                        # Store the individual actual dataframe
                        df_actual_file_hl = os.path.join(actual_folder, f"{file}")
                        df_actual_hl.to_excel(df_actual_file_hl, index=False)
                    elif file.endswith(f".csv"):
                        blob_client_hl = container_client_preprocess.get_blob_client("Actual/"+file)
                        content_ac_csv_hl = blob_client_hl.download_blob().readall()
                        df_actual_hl = pd.read_csv(io.BytesIO(content_ac_csv_hl))
                        # Store the individual actual dataframe
                        df_actual_file_hl = os.path.join(actual_folder, f"{file}")
                        df_actual_hl.to_csv(df_actual_file_hl, index=False)

                else:
                    if file.endswith(f".xlsx"):
                        blob_client_fl = container_client_preprocess.get_blob_client("Actual/"+file)
                        content_ac_excel_fl = blob_client_fl.download_blob().readall()
                        df_actual_fl = pd.read_excel(io.BytesIO(content_ac_excel_fl))
                        # Store the individual actual dataframe
                        df_actual_file_fl = os.path.join(actual_folder, f"{file}")
                        df_actual_fl.to_excel(df_actual_file_fl, index=False)

                    elif file.endswith(f".csv"):
                        blob_client_fl = container_client_preprocess.get_blob_client("Actual/"+file)
                        content_ac_csv_fl = blob_client_fl.download_blob().readall()
                        df_actual_fl = pd.read_csv(io.BytesIO(content_ac_csv_fl))
                        # Store the individual actual dataframe
                        df_actual_file_fl = os.path.join(actual_folder, f"{file}")
                        df_actual_fl.to_csv(df_actual_file_fl, index=False)

            self.logger.log(message=f"df_actual_hl:{str(df_actual_hl)}")
            self.logger.log(message=f"df_actual_fl:{str(df_actual_fl)}")

            # Replace NaN values with 9
            df_actual_hl = df_actual_hl.fillna(9)
            df_actual_fl = df_actual_fl.fillna(9)
            filtered_rows_actual_hl = df_actual_hl.loc[df_actual_hl["BA_ID_generated"].notnull()]
            filtered_rows_actual_fl = df_actual_fl.loc[df_actual_fl["BA_ID_generated"].notnull()]

            df_actual_list.append(df_actual_hl)
            df_actual_list.append(df_actual_fl)        

            # Extract the values of each column if present
            try:
                if "Uniform" in filtered_rows_actual_fl.columns:
                    uniform_actual.extend(filtered_rows_actual_fl["Uniform"].values.tolist())
            except KeyError:
                pass

            try:
                if "Shoes" in filtered_rows_actual_fl.columns:
                    shoes_actual.extend(filtered_rows_actual_fl["Shoes"].values.tolist())
            except KeyError:
                pass

            try:
                if "Blusher" in filtered_rows_actual_hl.columns:
                    blusher_actual.extend(filtered_rows_actual_hl["Blusher"].values.tolist())
            except KeyError:
                pass

            try:
                if "EyeLiner" in filtered_rows_actual_hl.columns:
                    eyeliner_actual.extend(filtered_rows_actual_hl["EyeLiner"].values.tolist())
            except KeyError:
                pass

            try:
                if "EyeShadow" in filtered_rows_actual_hl.columns:
                    eyeshadow_actual.extend(filtered_rows_actual_hl["EyeShadow"].values.tolist())
            except KeyError:
                pass

            try:
                if "Lipstick" in filtered_rows_actual_hl.columns:
                    lipstick_actual.extend(filtered_rows_actual_hl["Lipstick"].values.tolist())
            except KeyError:
                pass

            try:
                if "Shaved/Groomed Beard" in filtered_rows_actual_hl.columns:
                    shaved_beard_actual.extend(filtered_rows_actual_hl["Shaved/Groomed Beard"].values.tolist())
            except KeyError:
                pass

            try:
                if "Combed Hairs" in filtered_rows_actual_hl.columns:
                    combed_hairs_actual.extend(filtered_rows_actual_hl["Combed Hairs"].values.tolist())
            except KeyError:
                pass


            list_file_name_pred = []
            for blob in container_client_preprocess.list_blobs():
                if "predicted" in blob.name.lower():
                    file_name_pred = blob.name.split('/')[-1]
                    list_file_name_pred.append(file_name_pred)
                    dates_dynamic = []
                    # Split the file_name directly
                    date_str = file_name_pred.split('.')[0].split('_')[-1]
                    date = datetime.strptime(date_str, '%d-%m-%Y').date()
                    dates_dynamic.append(date)
                    latest_date_dynamic = max(dates_dynamic)
            file_dynamic_pred = {}
            for file in list_file_name_pred:
                if latest_date_dynamic.strftime("%d-%m-%Y") in file.lower():
                    folder_name = "Predicted"
                    if folder_name not in file_dynamic_pred:
                        file_dynamic_pred[folder_name] = [file]
                    else:
                        file_dynamic_pred[folder_name].append(file)
            self.logger.log(message=f"file_dynamic_pred:{str(file_dynamic_pred)}")

            # Read the content using pandas
            predicted_list = file_dynamic_pred['Predicted']
            for file in predicted_list:
                if "half" in file:
                    if file.endswith(f".xlsx"):
                        # Download the blob content
                        blob_client_hl = container_client_preprocess.get_blob_client("Predicted/"+file)
                        content_pred_excel_hl = blob_client_hl.download_blob().readall()
                        df_predicted_hl = pd.read_excel(io.BytesIO(content_pred_excel_hl))
                        # Store the individual actual dataframe
                        df_predicted_file_hl = os.path.join(predicted_folder, f"{file}")
                        df_predicted_hl.to_excel(df_predicted_file_hl, index=False)
                    elif file.endswith(f".csv"):
                        # Download the blob content
                        blob_client_hl = container_client_preprocess.get_blob_client("Predicted/"+file)
                        content_pred_excel_hl = blob_client_hl.download_blob().readall()
                        df_predicted_hl = pd.read_excel(io.BytesIO(content_pred_excel_hl))
                        # Store the individual actual dataframe
                        df_predicted_file_hl = os.path.join(predicted_folder, f"{file}")
                        df_predicted_hl.to_csv(df_predicted_file_hl, index=False)

                else:
                    if file.endswith(f".xlsx"):
                        blob_client_fl = container_client_preprocess.get_blob_client("Predicted/"+file)
                        content_pred_excel_fl = blob_client_fl.download_blob().readall()
                        df_predicted_fl = pd.read_excel(io.BytesIO(content_pred_excel_fl))
                        # Store the individual actual dataframe
                        df_predicted_file_fl = os.path.join(predicted_folder, f"{file}")
                        df_predicted_fl.to_excel(df_predicted_file_fl, index=False)

                    elif file.endswith(f".csv"):
                        blob_client_fl = container_client_preprocess.get_blob_client("Predicted/"+file)
                        content_pred_excel_fl = blob_client_fl.download_blob().readall()
                        df_predicted_fl = pd.read_excel(io.BytesIO(content_pred_excel_fl))
                        # Store the individual actual dataframe
                        df_predicted_file_fl = os.path.join(predicted_folder, f"{file}")
                        df_predicted_fl.to_csv(df_predicted_file_fl, index=False)

            self.logger.log(message=f"df_predicted_hl:{str(df_predicted_hl)}")
            self.logger.log(message=f"df_predicted_fl:{str(df_predicted_fl)}")

            # Replace NaN values with 10
            df_predicted_hl = df_predicted_hl.fillna(10)
            df_predicted_fl = df_predicted_fl.fillna(10)
            filtered_rows_predicted_hl = df_predicted_hl.loc[df_predicted_hl["BA_ID_generated"].notnull()]
            filtered_rows_predicted_fl = df_predicted_fl.loc[df_predicted_fl["BA_ID_generated"].notnull()]

            df_predicted_list.append(df_predicted_hl)
            df_predicted_list.append(df_predicted_fl)


            # Extract the values of each column if present
            try:
                if "Uniform" in filtered_rows_predicted_fl.columns:
                    uniform_pred.extend(filtered_rows_predicted_fl["Uniform"].values.tolist())
            except KeyError:
                pass

            try:
                if "Shoes" in filtered_rows_predicted_fl.columns:
                    shoes_pred.extend(filtered_rows_predicted_fl["Shoes"].values.tolist())
            except KeyError:
                pass

            try:
                if "Blusher" in filtered_rows_predicted_hl.columns:
                    blusher_pred.extend(filtered_rows_predicted_hl["Blusher"].values.tolist())
            except KeyError:
                pass

            try:
                if "EyeLiner" in filtered_rows_predicted_hl.columns:
                    eyeliner_pred.extend(filtered_rows_predicted_hl["EyeLiner"].values.tolist())
            except KeyError:
                pass

            try:
                if "EyeShadow" in filtered_rows_predicted_hl.columns:
                    eyeshadow_pred.extend(filtered_rows_predicted_hl["EyeShadow"].values.tolist())
            except KeyError:
                pass

            try:
                if "Lipstick" in filtered_rows_predicted_hl.columns:
                    lipstick_pred.extend(filtered_rows_predicted_hl["Lipstick"].values.tolist())
            except KeyError:
                pass

            try:
                if "Shaved/Groomed Beard" in filtered_rows_predicted_hl.columns:
                    shaved_beard_pred.extend(filtered_rows_predicted_hl["Shaved/Groomed Beard"].values.tolist())
            except KeyError:
                pass

            try:
                if "Combed Hairs" in filtered_rows_predicted_hl.columns:
                    combed_hairs_pred.extend(filtered_rows_predicted_hl["Combed Hairs"].values.tolist())
            except KeyError:
                pass


            # Count the total number of entries for each category
            uniform_actual_count = len(uniform_actual)
            uniform_pred_count = len(uniform_pred)
            shoes_actual_count = len(shoes_actual)
            shoes_pred_count = len(shoes_pred)
            blusher_actual_count = len(blusher_actual)
            blusher_pred_count = len(blusher_pred)
            eyeliner_actual_count = len(eyeliner_actual)
            eyeliner_pred_count = len(eyeliner_pred)
            eyeshadow_actual_count = len(eyeshadow_actual)
            eyeshadow_pred_count = len(eyeshadow_pred)
            lipstick_actual_count = len(lipstick_actual)
            lipstick_pred_count = len(lipstick_pred)
            shaved_beard_actual_count = len(shaved_beard_actual)
            shaved_beard_pred_count = len(shaved_beard_pred)
            combed_hairs_actual_count = len(combed_hairs_actual)
            combed_hairs_pred_count = len(combed_hairs_pred)

            # Store the counts in a DataFrame
            data = {
                'Category': ['uniform_actual', 'uniform_pred', 'shoes_actual', 'shoes_pred', 'blusher_actual', 'blusher_pred',
                             'eyeliner_actual', 'eyeliner_pred', 'eyeshadow_actual', 'eyeshadow_pred', 'lipstick_actual',
                             'lipstick_pred', 'shaved_beard_actual', 'shaved_beard_pred', 'combed_hairs_actual',
                             'combed_hairs_pred'],
                'Count': [uniform_actual_count, uniform_pred_count, shoes_actual_count, shoes_pred_count, blusher_actual_count,
                          blusher_pred_count, eyeliner_actual_count, eyeliner_pred_count, eyeshadow_actual_count,
                          eyeshadow_pred_count, lipstick_actual_count, lipstick_pred_count, shaved_beard_actual_count,
                          shaved_beard_pred_count, combed_hairs_actual_count, combed_hairs_pred_count]
            }
            df_counts = pd.DataFrame(data)

            # Save the counts to a CSV file
            counts_file = os.path.join(self.output_folder, "Total_counts_of_all_columns.csv")
            df_counts.to_csv(counts_file, index=False)        

            # Reshape the arrays to a suitable format
            self.y_true = np.concatenate([uniform_actual, shoes_actual, blusher_actual, eyeliner_actual, eyeshadow_actual, lipstick_actual, shaved_beard_actual, combed_hairs_actual])
            self.y_pred = np.concatenate([uniform_pred, shoes_pred, blusher_pred, eyeliner_pred, eyeshadow_pred, lipstick_pred, shaved_beard_pred, combed_hairs_pred])

            self.logger.log(message=f"self.y_true:{str(self.y_true)}")
            self.logger.log(message=f"self.y_pred:{str(self.y_pred)}")

            return self.y_true, self.y_pred
        except Exception as e:
            self.logger.log(message=f"read_files_from_container function is skipped and error is:{str(e)}")            

#########VISUAL_QUALITY_ASSESMENT##############
    def get_latest_month_year_folder(self, container_client):
        try:
            blobs = container_client.list_blobs()
            date_format = "%d-%m-%Y"
            dates = set()

            for blob in blobs:
                folder_name = blob.name.split('/')[0]
                try:
                    folder_date = datetime.strptime(folder_name, date_format).date()
                    dates.add(folder_date)
                except ValueError:
                    pass

            self.latest_date = max(dates)
            self.logger.log(message=f"latest_date.month:{str(self.latest_date.month)}")
            self.logger.log(message=f"latest_date.year:{str(self.latest_date.year)}")
            
            return self.latest_date.month, self.latest_date.year
        except Exception as e:
            self.logger.log(message=f"get_latest_month_year_folder function is skipped and error is:{str(e)}")
    

########FULL LENGTH AND HALF LENGTH IMAGES################
    def visual_quality(self):
        try:
            ##########Process FULL LENGTH images ##############################
            self.latest_month, self.latest_year = self.get_latest_month_year_folder(self.container_client_FL)
            for blob in self.container_client_FL.list_blobs():
                folder_name = blob.name.split('/')[0]
                folder_date = datetime.strptime(folder_name, "%d-%m-%Y").date()
                if folder_date.month == self.latest_month and folder_date.year == self.latest_year:
                    if blob.name.endswith((".jpg", ".jpeg", ".png")):
                        with tempfile.NamedTemporaryFile(delete=True) as temp_file:
                            blob_client = self.container_client_FL.get_blob_client(blob.name)
                            blob_client.download_blob().download_to_stream(temp_file)
                            temp_file.seek(0)
                            image_FL = Image.open(temp_file)
                            resolution = image_FL.size
                            size = blob.size
                            image_path_FL = f'=HYPERLINK("{self.container_client_FL.url}/{blob.name}")'
                            date = blob.name.split("/")[0]
                            image_rgb_FL = image_FL.convert('RGB')
                            brisque_score = BRISQUE().score(image_rgb_FL)
                            if brisque_score < self.threshold:
                                quality_assessment = "Good quality and clarity"
                            else:
                                quality_assessment = "Poor quality and clarity"
                            self.csv_writer_FL.writerow([date, blob.name, f"{resolution[0]} x {resolution[1]}", size, image_path_FL, brisque_score, quality_assessment])

            ########## Process HALF LENGTH images ######################################
            self.latest_month, self.latest_year = self.get_latest_month_year_folder(self.container_client_HL)
            for blob in self.container_client_HL.list_blobs():
                folder_name = blob.name.split('/')[0]
                folder_date = datetime.strptime(folder_name, "%d-%m-%Y").date()
                if folder_date.month == self.latest_month and folder_date.year == self.latest_year:
                    if blob.name.endswith((".jpg", ".jpeg", ".png")):
                        with tempfile.NamedTemporaryFile(delete=True) as temp_file:
                            blob_client = self.container_client_HL.get_blob_client(blob.name)
                            blob_client.download_blob().download_to_stream(temp_file)
                            temp_file.seek(0)
                            image_HL = Image.open(temp_file)
                            resolution = image_HL.size
                            size = blob.size
                            image_path_HL = f'=HYPERLINK("{self.container_client_HL.url}/{blob.name}")'
                            date = blob.name.split("/")[0]
                            image_rgb_HL = image_HL.convert('RGB')
                            brisque_score = BRISQUE().score(image_rgb_HL)
                            if brisque_score < self.threshold:
                                quality_assessment = "Good quality and clarity"
                            else:
                                quality_assessment = "Poor quality and clarity"
                            self.csv_writer_HL.writerow([date, blob.name, f"{resolution[0]} x {resolution[1]}", size, image_path_HL, brisque_score, quality_assessment])

            # Close the CSV files
            self.csv_file_FL.close()
            self.csv_file_HL.close()
            self.logger.log(message=f"Visual quality outputs are generated successfully.")
        except Exception as e:
            self.logger.log(message=f"Visual_quality function is skipped and error is: {str(e)}")
    
#########ACCURACY##############
    def calculate_accuracy(self):
        try:
            accuracy = accuracy_score(self.y_true, self.y_pred)
            return accuracy
        except Exception as e:
            self.logger.log(message=f"calculate_accuracy function is skipped and error is: {str(e)} and also check for y_true and y_pred count having the same length or not")
            return 0

#########PRECISION##############   
    def calculate_precision(self):
        try:
            precision = precision_score(self.y_true, self.y_pred,average='macro')
            return precision
        except Exception as e:
            self.logger.log(message=f"calculate_precision function is skipped and error is: {str(e)}  and also check for y_true and y_pred count having the same length or not")
            return 0

#########RECALL##############    
    def calculate_recall(self):
        try:
            recall = recall_score(self.y_true, self.y_pred,average='macro')
            return recall
        except Exception as e:
            self.logger.log(message=f"calculate_recall function is skipped and error is: {str(e)}  and also check for y_true and y_pred count having the same length or not")
            return 0
        
#########ROBUSTNESS##############   
    def calculate_robustness(self, precision, recall, accuracy):
        try:
            # Calculate F1-score
            f1_score = 2 * (precision * recall) / (precision + recall)
            # Calculate robustness
            robustness = (accuracy + f1_score) / 2


            self.logger.log(message=f"f1_score: {str(f1_score)}")
            self.logger.log(message=f"robustness: {str(robustness)}")
            # Return results
            return robustness
        except Exception as e:
            self.logger.log(message=f"calculate_recall function is skipped and error is: {str(e)} and also check for precision, recall and accuracy scores")
    
#########Error_rate##############    
    def calculate_error_rate(self):
        total_errors = 0
        try:
            total_samples = len(self.y_true)  # Assuming y_true and y_pred have the same length
            for true_label, pred_label in zip(self.y_true, self.y_pred):
                if true_label != pred_label:
                    total_errors += 1

                error_rate = total_errors / total_samples
            return error_rate
        except Exception as e:
            self.logger.log(message=f"calculate_error_rate function is skipped and error is: {str(e)} and also check for y_true and y_pred count having the same length or not")

#########Error_Analysis##############
    def conduct_error_analysis(self):
        try:
            true_positives = []
            true_negatives = []
            false_positives = []
            false_negatives = []
            NA_errors = []

            for true_label, pred_label in zip(self.y_true, self.y_pred):
                # Check if true_label and pred_label are valid (0 or 1)
                if true_label in [0, 1] and pred_label in [0, 1]:
                    if true_label == 1 and pred_label == 1:  # True positive
                        true_positives.append((true_label, pred_label))
                    elif true_label == 0 and pred_label == 0:  # True negative
                        true_negatives.append((true_label, pred_label))
                    elif true_label == 0 and pred_label == 1:  # False positive
                        false_positives.append((true_label, pred_label))
                    elif true_label == 1 and pred_label == 0:  # False negative
                        false_negatives.append((true_label, pred_label))
                else:
                    # Handle any label other than 0 or 1
                    NA_errors.append((true_label, pred_label))

            # Calculate the error rates
            total_samples = len(self.y_true)
            total_errors = len(false_positives) + len(false_negatives)

            self.logger.log(message=f"false_positives: {str(false_positives)}")
            self.logger.log(message=f"false_negatives: {str(false_negatives)}")

            # Calculate the error rates as percentages
            false_positive_rate = (len(false_positives) / total_samples) * 100
            false_negative_rate = (len(false_negatives) / total_samples) * 100
            true_negative_rate = (len(true_negatives) / total_samples) * 100
            true_positive_rate = (len(true_positives) / total_samples) * 100


            error_rate = (total_errors / total_samples) * 100

            NA_Error_rate = (len(NA_errors) / total_samples) * 100

            error_analysis_file = os.path.join(self.output_folder, "Error_Analysis_Suggestions&Improvements.csv")
            with open(error_analysis_file, "w", newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Sl.No", "Actual", "Prediction", "Combination", "Suggestion/Improvement"])

                index = 1
                if false_positive_rate > 0:
                    for true_label, pred_label in false_positives:
                        suggestions_pos = "Reduce false positives by adjusting the classification threshold or improving the model's specificity."
                        writer.writerow([index, true_label, pred_label, "False Positives", suggestions_pos])
                        index += 1
                if false_negative_rate > 0:
                    for true_label, pred_label in false_negatives:
                        suggestions_neg = "Reduce false negatives by fine-tuning the model's sensitivity or augmenting positive samples."
                        writer.writerow([index, true_label, pred_label, "False Negatives", suggestions_neg])
                        index += 1
                if error_rate == 0:
                    for true_label, pred_label in (true_positives+true_negatives):
                        suggestions_nill = "No errors found. The model is performing well."
                        writer.writerow([index, true_label, pred_label, "No Errors Found", suggestions_nill])
                        index += 1
                if NA_Error_rate != 0:
                    for true_label, pred_label in NA_errors:
                        suggestions_other = "These are invalid errors which has labels other than 0 or 1"
                        writer.writerow([index, true_label, pred_label, "NA Errors", suggestions_other])
                        index += 1

                writer.writerow([])
                writer.writerow(["False Positive Rate", false_positive_rate])
                writer.writerow(["False Negative Rate", false_negative_rate])
                writer.writerow(["True Negative Rate", true_negative_rate])
                writer.writerow(["True Positive Rate", true_positive_rate])
                writer.writerow(["NA Error Rate", NA_Error_rate])

            self.logger.log(message=f"error analysis is generated successfully.")
            return true_positives, true_negatives, false_positives, false_negatives, false_positive_rate, false_negative_rate
        except Exception as e:
            self.logger.log(message=f"conduct_error_analysis function is skipped and error is: {str(e)}")

#########Metrices Calculation##############
    def calculate_metrics(self):
        try:
            with open('config_model.json') as config_file:
                config_data = json.load(config_file)
            metrics = {}
            metrics['solution_name']=config_data['solution_name']
            metrics['solution_id']=config_data['solution_id']
            metrics['model_name']=config_data['model_name']
            metrics['model_type']=config_data['model_type']

            accuracy = self.calculate_accuracy() 
            metrics['metrics_name']='accuracy' 
            metrics['metrics_value']=accuracy
            metrics['metrics_percentage(%)']=accuracy*100
            df =pd.DataFrame([metrics])

            precision = self.calculate_precision()
            metrics['metrics_name']='precision' 
            metrics['metrics_value']=precision
            metrics['metrics_percentage(%)']=precision*100
            df1 = pd.DataFrame([metrics])
            df=df.append(df1,ignore_index=True)

            recall = self.calculate_recall()
            metrics['metrics_name']='recall' 
            metrics['metrics_value']=recall
            metrics['metrics_percentage(%)']=recall*100
            df2 = pd.DataFrame([metrics])
            df = df.append(df2,ignore_index=True)

            robustness = self.calculate_robustness(precision, recall, accuracy)
            metrics['metrics_name'] = 'robustness'
            metrics['metrics_value'] = robustness
            metrics['metrics_percentage(%)']=robustness*100
            df3 = pd.DataFrame([metrics])
            df = df.append(df3, ignore_index=True)

            error_rate = self.calculate_error_rate()
            metrics['metrics_name'] = 'error_rate'
            metrics['metrics_value'] = error_rate
            metrics['metrics_percentage(%)'] = error_rate * 100
            df4 = pd.DataFrame([metrics])
            df = df.append(df4, ignore_index=True)
            df_final = os.path.join(self.output_folder,"metrices_ba_grooming.csv")
            df.to_csv(df_final, index = False)

            self.logger.log(message=f"Metrices Calculation is completed.")
            return df
        except Exception as e:
            self.logger.log(message=f"calculate_metrices is skipped and error is: {str(e)}")

#########Uploading files to blob##############
    def upload_files_to_blob(self):
        try:
                        
            self.latest_month, self.latest_year = self.get_latest_month_year_folder(self.container_client_FL)

            # Rename and clone the folder
            renamed_folder = f"{self.output_folder}_{self.latest_month}-{self.latest_year}"

            # Delete the destination folder if it already exists
            if os.path.exists(renamed_folder):
                shutil.rmtree(renamed_folder)

            shutil.copytree(self.output_folder, renamed_folder)

            if os.path.exists(renamed_folder):
                for folder in os.walk(renamed_folder):
                    for file in folder[-1]:
                        try:
                            blob_path = os.path.join(folder[0].replace(os.getcwd() + '\\', ''), file)
                            container_client = self.blob_service_client.get_container_client(self.upload_files_to_container)

                            # Create the container if it doesn't exist
                            if not container_client.exists():
                                container_client.create_container()

                            blob_obj = container_client.get_blob_client(blob=blob_path)
                            with open(os.path.join(folder[0],file),mode = 'rb') as file_data:
                                blob_obj.upload_blob(file_data, overwrite = True)
                        except ResourceExistsError:
                            self.logger.log(message=f"Blob already exists and cannot be overwritened: {str(blob_path)}")
                            continue
            else:
                self.logger.log(message="Renamed folder does not exist.")

            self.logger.log(message=f"Renamed folder '{str(renamed_folder)}' uploaded successfully.")
        except Exception as e:
            self.logger.log(message=f"An error occurred in upload_files_to_blob function and error is: {str(e)}")
                    
###create an instance for Metrices calculator
calculator = CV_MetricsCalculator()

############Execute all functions####################
calculator.visual_quality()
calculator.read_files_from_container()
calculator.calculate_metrics()
calculator.conduct_error_analysis()
calculator.upload_files_to_blob()

