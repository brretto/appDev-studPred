from django.urls import path
from .views import SyncGoogleFormView, SurveyResponseListView, PredictionSummaryView # Add new views

urlpatterns = [
    path('sync-forms/', SyncGoogleFormView.as_view(), name='sync-google-forms'),
    path('responses/', SurveyResponseListView.as_view(), name='response-list'), # Add this
    path('summary/', PredictionSummaryView.as_view(), name='prediction-summary'), # Add this
]