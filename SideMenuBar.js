import React from 'react';
import './SideMenuBar.css';
//https://ibaslogic.com/how-to-add-hamburger-menu-in-react/ For navbar explanation
function SideMenuBar (props){
  return(<div className={`side-menu-bar ${props.show ? 'open' : ''}`}>
    <button className={`side-menu-bar-button ${props.show ? 'side-menu-bar.open' : 'side-menu-bar'}`} onClick={props.close}>X</button>
    <ul className="side-menu-bar__list">
        {props.name.map((item, index) => (
              <li key={index}>{item}</li>
            ))}
    </ul>
  </div>)
};

export default SideMenuBar;
