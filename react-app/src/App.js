import React, { Component } from "react";
import {fetchOptions} from "./utils";
import "./App.css";




class App extends Component {
  constructor() {
    super();
    this.state = {
      server_data: null,
      client_data: null,
    };
  }

  getServerData(){
    fetch("/get_server_info_json", fetchOptions("GET"))
    .then((response) => response.json())
    .then((res) => this.setState({server_data: res}));
  }

  getClientData(){
    fetch("/get_client_info_json", fetchOptions("GET"))
    .then((response) => response.json())
    .then((res) => this.setState({client_data: res}));
  }

  componentDidMount() {
    // fetch the data every 5 seconds
    this.interval = setInterval(() => {
      this.getServerData();
      this.getClientData();
    }, 5000); 
    // each fetch triggers a state update
  }

  componentDidUpdate(prevProps, prevState){
    console.log("State updated:", this.state);
  }

  componentWillUnmount(){
    clearInterval(this.interval);
  }
  
  render() {
    return (
      // The rest of the file is the same
      <div className="App">
        <header className="App-header">
          <p>{this.server_data}</p>
        </header>
      </div>
    );
  }
}

export default App;
