from django.db import models
from django.db import models
from django.conf import settings

class SearchHistory(models.Model):
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True
    )
    disease = models.CharField(max_length=255)
    matched_disease = models.CharField(max_length=255, blank=True)
    predicted_drugs = models.TextField(blank=True)   # comma-separated
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.disease} -> {self.matched_disease} ({self.created_at.strftime('%Y-%m-%d %H:%M')})"
