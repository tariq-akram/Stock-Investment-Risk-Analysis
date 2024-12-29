from django import forms

class StockAnalysisForm(forms.Form):
    RISK_CHOICES = [
        ('Low', 'Low'),
        ('Medium', 'Medium'),
        ('High', 'High')
    ]
    symbol = forms.CharField(label="Stock Symbol", max_length=10, required=True)
    risk_tolerance = forms.ChoiceField(label="Risk Tolerance", choices=RISK_CHOICES, required=True)
