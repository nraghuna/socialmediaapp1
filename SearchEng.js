import React,{useState,useEffect} from 'react';
import axios from 'axios';

function Search () {
    const [userInput,setUserInput]=useState('');
    const [prediction,setPrediction]= useState('');

    const handleUserInput= (event)=>{
        setUserInput(event.target.value)
    }
    const handleSendMessage= async()=>{
        if (userInput){   
            const data={
                userInput
            }    
        axios.post('http://127.0.0.1:5000/search',data).then(() => {
            getPrediction();
          })
          .catch((error) => {
            console.log('Error:', error);
          });
        }
    }

    
    const getPrediction = () => {
        axios
      .get('http://127.0.0.1:5000/search')
      .then((response) => {
        setPrediction(response.data);
      })
      .catch((error) => {
        console.log('Error:', error);
      });
    };

    useEffect(() => {
        getPrediction();
    }, []);

    return (
        <div>
          <div className="user-input">
            <input type="text" value={userInput} onChange={handleUserInput} />
            <button onClick={handleSendMessage}>Send</button>
          </div>
          <div>
          <h3>Prediction:</h3>
          <p>{JSON.stringify(prediction)}</p>
        </div>
        </div>
      );

}
export default Search