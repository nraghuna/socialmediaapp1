import React, { useEffect } from "react"
import Container from "./Container"

function Page(props) {
  useEffect(() => {
    document.title = `${props.title} | ReactApp`
    window.scrollTo(0, 0)
  }, [props.title])
  
        //const response= axios
        //.post("/signup", userData)
        //.then((response) => {
          //console.log(response);
        //})
        //.catch((error) => {
          //if (error.response) {
            //console.log(error.response);
            //console.log("server responded");
          //} else if (error.request) {
            //console.log("network error");
          //} else {
            //console.log(error);
          //}
        
        //});

  return <Container wide={props.wide}>{props.children}</Container>
}

export default Page