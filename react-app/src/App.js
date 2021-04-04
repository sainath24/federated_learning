import React, { Component } from "react";
import { fetchOptions } from "./utils";
import { ProgressBar, Card, CardDeck } from "react-bootstrap";
import Chip from "@material-ui/core/Chip";
import FiberManualRecordIcon from "@material-ui/icons/FiberManualRecord";
import { green, red } from "@material-ui/core/colors";
import AppBar from "@material-ui/core/AppBar";
import Toolbar from "@material-ui/core/Toolbar";
import Typography from "@material-ui/core/Typography";
import CircularProgress from "@material-ui/core/CircularProgress";
import "bootstrap/dist/css/bootstrap.min.css";
import "./App.css";

class App extends Component {
  constructor() {
    super();
    this.state = {
      server_data: null,
      client_data: null,
    };
  }

  getServerData() {
    fetch("/get_server_info_json", fetchOptions("GET"))
      .then((response) => response.json())
      .then((res) => this.setState({ server_data: res }))
      .catch((e) => console.log(e));
  }

  getClientData() {
    fetch("/get_client_info_json", fetchOptions("GET"))
      .then((response) => response.json())
      .then((res) => this.setState({ client_data: res }))
      .catch((e) => console.log(e));
  }

  componentDidMount() {
    // fetch the data every 5 seconds
    this.interval = setInterval(() => {
      this.getServerData();
      this.getClientData();
    }, 5000);
    // each fetch triggers a state update
  }

  componentDidUpdate(prevProps, prevState) {
    console.log("State updated:", this.state);
  }

  componentWillUnmount() {
    clearInterval(this.interval);
  }

  inProgressBar(key, completed, current, total) {
    return (
      <div>
        <ProgressBar
          key={key}
          animated
          striped
          now={completed}
          label={`${current} of ${total}`}
        />
      </div>
    );
  }

  completedProgressBar(key, completed, current, total) {
    return (
      <div>
        <ProgressBar
          key={key}
          animated
          striped
          variant="success"
          now={completed}
          label={`${current} of ${total}`}
        />
      </div>
    );
  }

  getInactiveChip() {
    return (
      <Chip
        variant="outlined"
        color="secondary"
        icon={<FiberManualRecordIcon style={{ color: red[500] }} />}
        label="Inactive"
      />
    );
  }

  getActiveChip() {
    return (
      <Chip
        style={{ color: green[500] }}
        variant="outlined"
        icon={<FiberManualRecordIcon style={{ color: green[500] }} />}
        label="Active"
      />
    );
  }

  renderClients() {
    const { client_data } = this.state;

    return (
      <div className="client">
        <CardDeck>
          {Object.entries(client_data).map(([key, value]) => {
            console.log(key, client_data[key]);
            const current_epoch = value["current_epoch"] + 1;
            const total_epochs = value["total_epochs"];
            const completed = (current_epoch / total_epochs) * 100;
            const condition = value["condition"];
            const status = value["status"];
            const exception = value["exception"];
            return (
              <div>
                <Card className="mb-4" style={{ width: "18rem" }}>
                  <Card.Img
                    variant="top"
                    src="hospital.png"
                    width={100}
                    height={250}
                  />
                  <Card.Body>
                    <Card.Title>{key}</Card.Title>
                    {condition === "Alive" ||condition === 'Completed'
                      ? this.getActiveChip()
                      : this.getInactiveChip()}
                    {condition === "Dead"? <div class='error'><Card.Text>{exception}</Card.Text></div> : <Card.Text></Card.Text>}
                    <Card.Text>{status}</Card.Text>
                    {completed === 100
                      ? this.completedProgressBar(
                          key,
                          completed,
                          current_epoch,
                          total_epochs
                        )
                      : this.inProgressBar(
                          key,
                          completed,
                          current_epoch,
                          total_epochs
                        )}
                  </Card.Body>
                </Card>
              </div>
            );
          })}
        </CardDeck>
      </div>
    );
  }

  renderServer() {
    const { server_data } = this.state;
    const key = "server";
    const total_rounds = server_data["fl_rounds"];
    let fl_rounds_left = server_data["fl_rounds_left"];
    if ( fl_rounds_left < 0 ) {
      fl_rounds_left = 0;
    }
    const completed = ((total_rounds - fl_rounds_left) / total_rounds) * 100;
    return (
      <div className="server">
        <Card style={{ width: "18rem" }}>
          <Card.Img variant="top" src="server.png" width={250} height={300} />
          <Card.Body>
            <Card.Title>Server</Card.Title>
            {completed === 100
              ? this.completedProgressBar(
                  key + "_completed",
                  completed,
                  total_rounds - fl_rounds_left,
                  total_rounds
                )
              : this.inProgressBar(
                  key + "_inProgress",
                  completed,
                  total_rounds - fl_rounds_left,
                  total_rounds
                )}
          </Card.Body>
        </Card>
      </div>
    );
  }

  renderLoadingScreen() {
    return (
      <div style={{ diplay: "flex" }}>
        <CircularProgress color="primary" />
      </div>
    );
  }
  render() {
    return (
      // The rest of the file is the same
      <div className="App">
        <header className="App-header">
          <AppBar>
            <Toolbar>
              <Typography variant="h6">Federated Learning UI</Typography>
            </Toolbar>
          </AppBar>
          {this.state.server_data === null && this.state.client_data === null
            ? this.renderLoadingScreen()
            : null}
          {this.state.server_data ? this.renderServer() : null}
          {this.state.client_data ? this.renderClients() : null}
        </header>
      </div>
    );
  }
}

export default App;
