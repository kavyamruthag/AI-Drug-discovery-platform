from django.urls import path
from . import views
from .drug_probability_views import drug_predictor


urlpatterns = [
    path("", views.home_view, name="home"),
    path("register/", views.register_view, name="register"),
    path("login/", views.login_view, name="login"),
    path("logout/", views.logout_view, name="logout"),
    path("dashboard/", views.dashboard_view, name="dashboard"),
    path("profile/", views.profile_view, name="profile"),
    path("profile/edit/", views.edit_profile_view, name="edit_profile"),
    #path("profile/change-password/", views.change_password_view, name='change_password'), 
    path('profile/change-password/', views.change_password_view, name='change_password'),
    path('search/', views.search_view, name="search"),
    path("admin_dashboard/", views.admin_dashboard, name="admin_dashboard"),
    path("admin_login/", views.admin_login, name="admin_login"),
    path('analytics/', views.analytics_view, name='analytics'), 
    path('save_search/', views.save_search, name='save_search'),
    path('reports/', views.reports_view, name='reports'),
    path("download/<str:filename>/", views.download_csv, name="download_csv"),
    #path('download_report/', views.download_report, name='download_report'),
    path('resources/', views.resources_page, name='resources'),
    path('view/<str:filename>/', views.view_file, name='view_file'),
    #path('drug_predictor/', views.drug_predictor, name='drug_predictor'),
    path("disease_predictor/", views.disease_deficiency_predictor, name="disease_predictor"),
    #path("drug_dosage/",views.drug_dosage,name="drug_dosage"),
   path("drug_dosage/", views.drug_dosage_view, name="dosage"),
    #path("molecule/",views.index,name="molecule"),
    path("index/",views.index,name="index"),
    path("features/",views.predict_features,name='features'),
    path("drug_predictor/", drug_predictor, name="drug_predictor"),

]
