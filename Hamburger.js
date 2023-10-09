import React from 'react'
import './HamburgerButton.css'

function HamburgerButton(props){
   return( <button className="hamburger-button" onClick={props.onClick}>
        <span className="hamburger-button__line"></span>
        <span className="hamburger-button__line"></span>
        <span className="hamburger-button__line"></span>
        <span className="hamburger-button__line"></span>
    </button>
   )
}

export default HamburgerButton;