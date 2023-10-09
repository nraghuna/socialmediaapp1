import React, {useState} from "react";
import { useNavigate} from "react-router-dom";

function Signup(){
    const [user,setUser]=   useState({
            username: "",
            email:"",
            password:""
    })
    const navigator = useNavigate();


    const handleChange = (e) =>{
        const value = e.target.value;
         setUser({
            ...user,
            [e.target.name]: value
    });
    }

    const Submit= async (e)=>{
      e.preventDefault();
      const userData = {
        username:user.username,
        email: user.email,
        password: user.password
      };
        try {
          const response = await fetch('/signup', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json'
            },
            body: JSON.stringify({
              userData
            })
          });
    
          const data = await response.json();
    
          if (response.ok) {
            //window.location.href = '/Profile';
            navigator('/signups');
          } else {
            console.log(data.message);
          }
        } catch (error) {
          console.log('Something went wrong. Please try again later.');
        }
    
        console.log(JSON.stringify(userData));
        navigator('/signups');
      }
    
    return(
        <div>
        <form onSubmit={Submit}>
        <div>
            <label className="label">username</label>
          <input onChange={(e) => handleChange(e)} type="username"
            name="username" defaultValue={user.username} />
        </div>
        <div>
            <label className="label">email</label>
          <input onChange={(e) => handleChange(e)} className="input" type="email"
            name="email" defaultValue={user.email} />
        </div>
        <div>
            <label className="label">password</label>
          <input onChange={(e) => handleChange(e)} className="input" type="password"
            name="password" defaultValue={user.password} />
        </div>
        <button type="submit">Submit</button>
        </form>
        </div>
    )
}

export default Signup