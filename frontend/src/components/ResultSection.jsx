import React from 'react';
import { Bar } from 'react-chartjs-2';
import {
    Chart as ChartJS,
    CategoryScale,
    LinearScale,
    BarElement,
    Title,
    Tooltip,
    Legend
} from 'chart.js';

ChartJS.register(
    CategoryScale,
    LinearScale,
    BarElement,
    Title,
    Tooltip,
    Legend
);

function ResultSection({ data }) {
    if (!data) return null;

    return (
        <div style={styles.container}>
            <h2>Results for: {data.brand}</h2>

            <div style={styles.summary}>
                <p><strong>Mentions:</strong> {data.mentions}</p>
                <p><strong>Positive:</strong> {data.positive}</p>
                <p><strong>Negative:</strong> {data.negative}</p>
                <p><strong>Neutral:</strong> {data.neutral}</p>
            </div>

            <Bar
                data={{
                    labels: ['Positive', 'Negative', 'Neutral'],
                    datasets: [{
                        label: 'Sentiment',
                        data: [data.positive, data.negative, data.neutral],
                        backgroundColor: ['green', 'red', 'gray'],
                    }]
                }}
            />
        </div>
    );
}

const styles = {
    container: { marginTop: '30px', textAlign: 'center' },
    summary: { marginBottom: '20px' }
};

export default ResultSection;
