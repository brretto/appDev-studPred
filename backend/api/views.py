from django.shortcuts import render

# Create your views here.

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .models import SurveyResponse, Prediction
import gforms_client # CORRECTED IMPORT
from ml.predictor import Predictor
from ml.recommender import get_recommendation

# It's efficient to load the model once when the server starts
# rather than loading it on every API call.
predictor = Predictor()

class SyncGoogleFormView(APIView):
    """
    An API endpoint to sync responses from a Google Form,
    run predictions, and save the results to the database.
    """
    def post(self, request, *args, **kwargs):
        # IMPORTANT: Replace with your actual Form ID from the URL
        # e.g., "1abcdefg-HIJKLMN_o-pqrstuvwxyz"
        FORM_ID = "1qhh7AUwCahfPeyLR2W96Q1tBfls5h6PpQIDFj7gNBtc" 
        
        print("Fetching new responses from Google Forms...")
        raw_responses = gforms_client.get_form_responses(FORM_ID)

        
        
        if raw_responses is None:
            return Response(
                {"error": "Failed to fetch data from Google Forms API. Check credentials and form sharing permissions."},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
            
        parsed_responses = gforms_client.parse_responses(raw_responses)
        
        new_responses_count = 0
        for data in parsed_responses:
            # This is a simple check to avoid creating duplicate entries.
            # A more robust solution might check based on a unique identifier
            # from the form if one exists.
            if SurveyResponse.objects.filter(
                gender=data.get('Gender'),
                attendance_rate=data.get('AttendanceRate'),
                study_hours_per_week=data.get('StudyHoursPerWeek'),
                previous_grade=data.get('PreviousGrade'),
                extracurricular_activities=data.get('ExtracurricularActivities', '').lower() == 'yes',          
                parental_support=data.get('ParentalSupport'),
                online_classes_taken=data.get('Online Classes Taken')
                
                # Add more fields to make the check more unique
            ).exists():
                continue # Skip if this response seems to be a duplicate

            # 1. Save the Survey Response to the database
            response_obj = SurveyResponse.objects.create(
                gender=data.get('Gender'),
                attendance_rate=data.get('AttendanceRate'),
                study_hours_per_week=data.get('StudyHoursPerWeek'),
                previous_grade=data.get('PreviousGrade'),
                extracurricular_activities='yes' in str(data.get('ExtracurricularActivities', '')).lower(),
                parental_support=data.get('ParentalSupport'),
                online_classes_taken=int(data.get('Online Classes Taken', 0))
            )

            # 2. Run the prediction
            result, confidence = predictor.predict(data)
            
            # 3. Get a recommendation
            recommendation = get_recommendation(data, result)
            
            # 4. Save the Prediction and link it to the response
            Prediction.objects.create(
                survey_response=response_obj,
                predicted_result=result,
                confidence_score=confidence,
                recommendation=recommendation
            )
            new_responses_count += 1

        return Response({
            "message": f"Sync complete. Processed {new_responses_count} new responses.",
            "total_fetched": len(parsed_responses)
        }, status=status.HTTP_200_OK)
