import React, { useState } from 'react';
import SearchBar from '../components/SearchBar';
import ResultSection from '../components/ResultSection';

// Dummy data
const dummy = {
    brand: 'Zomato',
    mentions: 130,
    positive: 75,
    negative: 30,
    neutral: 25,
};

function Home() {
    const [result, setResult] = useState(null);

    const handleSearch = (brand) => {
        console.log('Searching for:', brand);
        // simulate API call (for minor version)
        setTimeout(() => {
            setResult({ ...dummy, brand }); // in real version, fetch from backend
        }, 500);
    };

    return (
        <div>
            <h1 style={{ textAlign: 'center' }}>Brand Monitoring Tool</h1>
            <SearchBar onSearch={handleSearch} />
            <ResultSection data={result} />
        </div>
    );
}

export default Home;
