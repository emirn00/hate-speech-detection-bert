import { Component } from '@angular/core';
import { HttpClient } from '@angular/common/http';

interface PredictResponse {
  text: string;
  prediction: string;
  confidence: number;
  model_used: string;
}

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent {
  title = 'Hate Speech Detection';
  inputText: string = '';
  result: PredictResponse | null = null;
  loading: boolean = false;
  error: string | null = null;

  constructor(private http: HttpClient) {}

  predict() {
    if (!this.inputText.trim()) return;

    this.loading = true;
    this.result = null;
    this.error = null;

    this.http.post<PredictResponse>('http://localhost:8000/predict', {
      text: this.inputText
    }).subscribe({
      next: (res) => {
        this.result = res;
        this.loading = false;
      },
      error: (err) => {
        console.error(err);
        this.error = "Could not connect to the backend API. Make sure FastAPI is running.";
        this.loading = false;
      }
    });
  }
}
