import React, { useState } from 'react';
import axios from 'axios';

function Search () {
    const [query,setQuery]=useState('');
    const [results,setResults]=useState([]);

    const handleInputChange = (event) => {
        setQuery(event.target.value);
      };



      axios
    .post('http://127.0.0.1:5000/searches', queries)
    .then(() => {
      //setSelectedOrders([])
      handleSearch();
    })
    .catch((error) => {
      console.log('Error:', error);
    });

    const queries = {
        query
      };

    const handleSearch = () => {
        axios
      .get('http://127.0.0.1:5000/searches',queries)
      .then((response) => {
        setResults(response.data);
      })
      .catch((error) => {
        console.log('Error:', error);
      });
    };

      //const handleSearch = async () => {
        //try {
         // const response = await axios.get('/searches', //{
            //params: { query },
         // });
         // setResults(response.data);
       // } catch (error) {
         // console.error('Error searching data:', error);
        //}
      //};

      
    useEffect(() => {
        handleSearch();
    }, []);



      return (
        <div>
          <input type="text" value={query} onChange={handleInputChange} />
          <button onClick={handleSearch}>Search</button>
          <ul>
            {results.map((result) => (
              <li key={result.id}>{result.name}</li>
            ))}
          </ul>
        </div>
      );
      
    };

export default Search