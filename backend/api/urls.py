from django.urls import path
from .views import SyncGoogleFormView

urlpatterns = [
    path('sync-forms/', SyncGoogleFormView.as_view(), name='sync-google-forms'),
]