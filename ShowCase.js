import React,{useState,useEffect} from 'react'
import axios from 'axios'
import './ShowCase.css';
//https://www.pluralsight.com/guides/how-to-use-multiline-text-area-in-reactjs
//https://blog.logrocket.com/how-to-use-axios-post-requests/
//https://stackoverflow.com/questions/46539480/react-clearing-an-input-value-after-form-submit
//https://upmostly.com/tutorials/how-to-post-requests-react
//https://medium.com/nerd-for-tech/fetching-api-using-useeffect-hook-in-react-js-7b9b34e427ca

function ShowCase(){
    const [post,setpost]= useState([])
    const [postcontent,setpostcontent]= useState('')
    useEffect(()=>{
        axios.get('http://127.0.0.1:5000/posts')
    .then((response) => {
        setpost(response.data) 
    })
    },[])
    
    useEffect(() => {
        const mockUser = [
            [
            "hwwhhwhw",
            "I love the Quueen"
            ],
            [
                "cool"
            ],
            [
                "cosssssssssssssssss"
            ],
            [
                "cosssssssssssssssss"
            ]
        ];
        setpost(mockUser);
      }, []);

    const handleSubmit = async () => {
        if (postcontent){
        await axios.post('http://127.0.0.1:5000/posts', {
            posts:postcontent})
    .then((response) => {
            setpost([
                ...post,
                response.data
            ]);
        setpostcontent('');
    });
    }
    }

    const handleChange=(event)=>{
        setpostcontent(
            event.target.value)
    }

    return(
        <div>
            <h1>ShowCase</h1>
            <form onSubmit={handleSubmit}>
                <textarea value={postcontent} onChange={handleChange} />
                <button type="submit">Submit</button>
            </form>
            {post.map((posts) => (
            //<div key={posts}>
            <div className ='posting-textbox' key={posts}>
            <p>{posts}</p>
            <div className='like-button'></div>
            <img src="src\marie.PNG" alt="Girl in a jacket" width="500" height="600"/>
            <div className="comment-section">
                        <textarea placeholder="Add a comment"></textarea>
                        <button>Submit Comment</button>
                        </div>
                        </div>
            ))}
        </div>
    )
//<div className="comment-box"></div>
}
export default ShowCase