// frontend/src/App.js

import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Bar, Pie } from 'react-chartjs-2';
import { Chart as ChartJS, ArcElement, Tooltip, Legend, CategoryScale, LinearScale, BarElement, Title } from 'chart.js';
import './App.css'; // We'll create this file for styling

// Register the components from Chart.js
ChartJS.register(ArcElement, Tooltip, Legend, CategoryScale, LinearScale, BarElement, Title);

function App() {
    const [activeTab, setActiveTab] = useState('predictions');
    const [responses, setResponses] = useState([]);
    const [summary, setSummary] = useState({ pass_count: 0, fail_count: 0, total_count: 0 });
    const [loading, setLoading] = useState(true);

    // Fetch data from Django API when the component mounts
    useEffect(() => {
        const fetchData = async () => {
            try {
                setLoading(true);
                const responsesRes = await axios.get('/api/responses/');
                const summaryRes = await axios.get('/api/summary/');
                setResponses(responsesRes.data);
                setSummary(summaryRes.data);
            } catch (error) {
                console.error("Failed to fetch data:", error);
                alert("Could not connect to the backend. Is the Django server running?");
            } finally {
                setLoading(false);
            }
        };
        fetchData();
    }, []);

    const handleSync = async () => {
        if (window.confirm("Are you sure you want to sync with Google Forms? This may take a moment.")) {
            try {
                alert("Syncing... please wait.");
                const res = await axios.post('/api/sync-forms/');
                alert(res.data.message);
                // Refresh data after sync
                window.location.reload();
            } catch (error) {
                console.error("Sync failed:", error);
                alert("Sync failed. Check the Django server console for errors.");
            }
        }
    };
    
    // --- Render Functions for Tabs ---

    const renderPredictionsTab = () => (
        <div>
            <h2>Prediction Dashboard</h2>
            {loading ? <p>Loading charts...</p> :
                <div className="charts-container">
                    <div className="chart">
                        <h3>Prediction Distribution</h3>
                        <Pie data={{
                            labels: ['PASS', 'FAIL'],
                            datasets: [{
                                data: [summary.pass_count, summary.fail_count],
                                backgroundColor: ['#4CAF50', '#F44336'],
                            }]
                        }} />
                    </div>
                    <div className="chart">
                        <h3>Prediction Counts</h3>
                        <Bar data={{
                            labels: ['PASS', 'FAIL'],
                            datasets: [{
                                label: '# of Students',
                                data: [summary.pass_count, summary.fail_count],
                                backgroundColor: ['#4CAF50', '#F44336'],
                            }]
                        }} options={{ responsive: true, plugins: { legend: { position: 'top' }, title: { display: true, text: 'Student Predictions' } } }} />
                    </div>
                </div>
            }
        </div>
    );

    const renderResponsesTab = () => (
        <div>
            <h2>Student Responses & Predictions</h2>
            {loading ? <p>Loading responses...</p> :
                <div className="table-container">
                    <table>
                        <thead>
                            <tr>
                                <th>ID</th>
                                <th>Previous Grade</th>
                                <th>Attendance (%)</th>
                                <th>Study Hours/Week</th>
                                <th>Prediction</th>
                                <th>Confidence</th>
                                <th>Recommendation</th>
                            </tr>
                        </thead>
                        <tbody>
                            {responses.map(res => (
                                <tr key={res.id}>
                                    <td>{res.id}</td>
                                    <td>{res.previous_grade}</td>
                                    <td>{res.attendance_rate}</td>
                                    <td>{res.study_hours_per_week}</td>
                                    <td className={res.prediction?.predicted_result === 'PASS' ? 'pass' : 'fail'}>
                                        {res.prediction?.predicted_result || 'N/A'}
                                    </td>
                                    <td>{res.prediction ? `${(res.prediction.confidence_score * 100).toFixed(1)}%` : 'N/A'}</td>
                                    <td>{res.prediction?.recommendation || 'N/A'}</td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            }
        </div>
    );

    const renderQuestionsTab = () => (
        <div>
            <h2>Questions</h2>
            <p>This application syncs directly with a Google Form. To add a new student, please fill out the form using the link below.</p>
            <a href="https://docs.google.com/forms/d/1qhh7AUwCahfPeyLR2W96Q1tBfls5h6PpQIDFj7gNBtc/viewform" target="_blank" rel="noopener noreferrer">
                Go to Google Form
            </a>
            <p className="note">Remember to replace the link above with your actual form's public URL.</p>
        </div>
    );

    const renderSettingsTab = () => (
        <div>
            <h2>Settings</h2>
            <button onClick={handleSync}>Sync with Google Forms</button>
            <p className="note">This will fetch the latest responses from your connected Google Form and update the database.</p>
        </div>
    );

    return (
        <div className="App">
            <header className="App-header">
                <h1>🎓 Student Success Predictor</h1>
                <p> A tool for predicting student success. (Pass/Fail) </p>
                <br></br>
                <br></br>
                <br></br>
                <br></br>
                <p> @Lagare Group </p>
            </header>
            <nav className="App-nav">
                <button onClick={() => setActiveTab('predictions')} className={activeTab === 'predictions' ? 'active' : ''}>Predictions</button>
                <button onClick={() => setActiveTab('responses')} className={activeTab === 'responses' ? 'active' : ''}>Responses</button>
                <button onClick={() => setActiveTab('questions')} className={activeTab === 'questions' ? 'active' : ''}>Questions</button>
                <button onClick={() => setActiveTab('settings')} className={activeTab === 'settings' ? 'active' : ''}>Settings</button>
            </nav>
            <main className="App-main">
                {activeTab === 'predictions' && renderPredictionsTab()}
                {activeTab === 'responses' && renderResponsesTab()}
                {activeTab === 'questions' && renderQuestionsTab()}
                {activeTab === 'settings' && renderSettingsTab()}
            </main>
        </div>
    );
}

export default App;