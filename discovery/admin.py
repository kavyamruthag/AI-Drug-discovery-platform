from django.contrib import admin

# Register your models here.
from django.contrib import admin
from .models import SearchHistory

@admin.register(SearchHistory)
class SearchHistoryAdmin(admin.ModelAdmin):
    list_display = ("disease", "matched_disease", "predicted_drugs", "user", "created_at")
    list_filter = ("created_at", "user")
    search_fields = ("disease", "matched_disease", "predicted_drugs", "user__username")
