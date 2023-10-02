function Chat(){
    const [convertdata,setconvertdata]= useState([])
    
    useEffect(()=>{
        async function fetchdata(){
            try{
                const response= await axios.get('http://127.0.0.1:5000/chatbot')
                const datas= await response.data
                setconvertdata(datas)
            }
            catch(error){
                console.error('Error fetching rating matrix:', error)
            }
        }
        fetchdata();
    },[])
    return(
        <div>
            <h2>Matching Data</h2>
            {convertdata.map((convert, index) => {
        return (
          <tr>
              <li> {convert} </li>;
          </tr>
        );
      })}
        </div>
    )
    }
    export default Chat