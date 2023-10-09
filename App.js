import './App.css';
import Signup from './signup';
import React,{useState,useEffect} from "react";

import {Routes,Route,useLocation} from "react-router-dom";
import Home from './Home';
import About from './About';
import Navbar from "./Navbar";
import ItemList from './ItemList';
import ProfilePage from './Profile';
import RatingMatrix from './RatingMatrix';
import RestaurantData from './Restaurantdata';
import MyForm from './MyForm';
import User from './User';
import Investor from './Investor';
import VideoUpload from './VideoUpload';
import ShowCaseUsercomments from './showcaseusercomments7';
//import ShowCasesusers from './Showcasesusers'
//import ShowCaseUsercomments from './showcaseusercomments2';
//import ShowCaseUsercomments from './Showcaseusercomments3';
//import ShowCasesss from './ShowCases';
import Chat from './chatbot';
import OrderSelection from './OrderSelection';
import ProductSelector from './ProductSelector'
//import DecisionTree from './decisiontree';
import Image from './Image';
import Imagejpg from './Imagejpg';
import ImageDisplay from './ImageDisplay';
import Imagesjpg from './Imagesjpg';
import UsersProfile from './UsersProfile'
import InvestorsProfile from './InvestorsProfile';
import Convert from './Convert';
import Search from './SearchEng';
import Recommedations from './Recommendation';
import Lines from './Liness';
import Posts from './Posts';
import Excel from './Excelworkbook';
import Login from './Login';
import Register from './Register';
import Upload from './docx';
import JobDescriptionGenerator from './jobdescription';
import ResumeGenerator from './ResumeGenerator';
import ResumeUploader from './ResumeUpload';
import Userdb from './Userdb';
import Hmanagerdb from './Hmanagers';
import Appointmentdb from './Appointment';
import AppointmentList from './AppointmentList';
import Automation from './Automation';
import Addappointments from './AddAppointment';
import CourseList from './CourseList';
import Coursemodel from './CourseApi';
//import Chat from './Chat';
function App() {
 
const datas= [{"active":{"label":"Active","value":"12"},"automatic":{"label":"Automatic","value":"8"},"waiting":{"label":"Waiting","value":"1"},"manual":{"label":"Manual","value":"3"}}]
const items = Object.keys(datas[0]).map(key => {
  return Object.values(datas[0][key]);
}).reduce((acc, val) => {
  return acc.concat(val);
}, []);
const [showNavbar, setShowNavbar] = useState(true);
const location = useLocation();
useEffect(() => {
    setShowNavbar(location.pathname === "/");
  }, [location]);
return(   

<div className="App">
{showNavbar && <Navbar />}
<Routes>
      <Route path="/" element={<Home/>} />
      <Route path="/about" element= {<About showNavbar={false}/>}/>
      <Route path="/signup" element={<Signup />} />
      <Route path="/items" element={<ItemList name = {items} />} />
      <Route path="/signups" element={<Coursemodel/>} />
</Routes>
</div>
);
}

export default App;
