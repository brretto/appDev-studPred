import os
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build

# Define the path to your credentials and the required scopes
CREDENTIALS_FILE = os.path.join(os.path.dirname(__file__), 'gcp_credentials.json')
SCOPES = ["https://www.googleapis.com/auth/forms.responses.readonly"]

def get_form_responses(form_id):
    """
    Connects to the Google Forms API and fetches all responses for a given form.

    Args:
        form_id (str): The ID of the Google Form.

    Returns:
        list: A list of response data, or None if an error occurs.
    """
    try:
        # Authenticate using the service account
        creds = Credentials.from_service_account_file(CREDENTIALS_FILE, scopes=SCOPES)
        service = build('forms', 'v1', credentials=creds)

        # Retrieve the form responses
        result = service.forms().responses().list(formId=form_id).execute()

        return result.get('responses', [])
    except Exception as e:
        print(f"An error occurred while fetching form responses: {e}")
        return None

def parse_responses(responses):
    """
    Parses the raw response data into a clean list of dictionaries.
    NOTE: This function needs to be customized to match your form's questions.
    """
    parsed_data = []
    if not responses:
        return parsed_data

    # --- IMPORTANT: You MUST customize this part ---
    # The 'questionId' values are unique to your form. You can find them by
    # inspecting the output of get_form_responses() for a sample response.
    QUESTION_MAPPING = {
        '279a4d82': 'Gender',
        '21be2a88': 'AttendanceRate',
        '7c69dd89': 'StudyHoursPerWeek',
        '0607f45f': 'PreviousGrade',
        '43db3521': 'ExtracurricularActivities',
        '5bfcb0cb': 'ParentalSupport',
        '54f9094c': 'Online Classes Taken',
        '289a6019': 'Name',
        # ... add all your other question IDs here
    }

    for response in responses:
        student_data = {}
        answers = response.get('answers', {})
        for question_id, answer_obj in answers.items():
            if question_id in QUESTION_MAPPING:
                # Extract the text value from the answer
                value = answer_obj['textAnswers']['answers'][0]['value']
                key = QUESTION_MAPPING[question_id]
                student_data[key] = value

        if student_data: # Only add if we parsed some data
            parsed_data.append(student_data)

    return parsed_data