import React, { useState } from 'react';

function SearchBar({ onSearch }) {
    const [brand, setBrand] = useState('');

    const handleSubmit = (e) => {
        e.preventDefault();
        if (brand.trim()) {
            onSearch(brand.trim());
        }
    };

    return (
        <form onSubmit={handleSubmit} style={styles.form}>
            <input
                type="text"
                placeholder="Enter brand name..."
                value={brand}
                onChange={(e) => setBrand(e.target.value)}
                style={styles.input}
            />
            <button type="submit" style={styles.button}>Search</button>
        </form>
    );
}

const styles = {
    form: { display: 'flex', gap: '10px', marginTop: '20px', justifyContent: 'center' },
    input: { padding: '10px', width: '300px', borderRadius: '5px', border: '1px solid gray' },
    button: { padding: '10px 20px', border: 'none', borderRadius: '5px', background: '#007bff', color: 'white', cursor: 'pointer' }
};

export default SearchBar;
