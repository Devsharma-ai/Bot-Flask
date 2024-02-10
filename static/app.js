class Chatbox {
  constructor() {
    this.args = {
      openButton: document.querySelector(".chatbox__button button"),
      chatBox: document.querySelector(".chatbox__support"),
      sendButton: document.querySelector(".send__button"),
    };
    this.state = false;
    this.message = [];
  }

  display() {
    const { openButton, chatBox, sendButton } = this.args;
    if (openButton && chatBox && sendButton) {
      openButton.addEventListener("click", () => this.toggleState(chatBox));
      sendButton.addEventListener("click", () => this.onSendButton(chatBox));

      const inputNode = chatBox.querySelector('input[type="text"]');
      if (inputNode) {
        inputNode.addEventListener("keyup", (event) => {
          if (event.key === "Enter") {
            this.onSendButton(chatBox);
          }
        });
      }
    } else {
      console.error("One or more elements not found.");
    }
  }

  toggleState(chatbox) {
    this.state = !this.state;
    if (this.state) {
      chatbox.classList.add("chatbox--active");
    } else {
      chatbox.classList.remove("chatbox--active");
    }
  }

  onSendButton(chatbox) {
    const textField = chatbox.querySelector('input[type="text"]');
    const text = textField.value.trim();
    if (!text) {
      return;
    }
    const messageObj = { name: "User", message: text };
    this.message.push(messageObj);
    // Assuming $SCRIPT_ROOT is defined elsewhere
    fetch($SCRIPT_ROOT + "/predict", {
      method: "POST",
      body: JSON.stringify({ message: text }),
      mode: "cors",
      headers: {
        "Content-Type": "application/json",
      },
    })
      .then((response) => response.json())
      .then((data) => {
        const msgObj = { name: "Sam", message: data.answer };
        this.message.push(msgObj);
        console.log({ chatbox });
        this.updateChatText(chatbox);
        textField.value = "";
      })
      .catch((error) => {
        console.error("Error:", error);
        this.updateChatText(chatbox);
        textField.value = "";
      });
  }

  updateChatText(chatbox) {
    var html = "";
    this.message
      .slice()
      .reverse()
      .forEach(function (item, index) {
        console.log(item.name);
        if (item.name === "Sam") {
          html +=
            '<div class="messages__item messages__item--operator">' +
            item.message +
            "</div>";
        } else {
          html +=
            '<div class="messages__item messages__item--visitor">' +
            item.message +
            "</div>";
        }
      });

    const chatmessage = chatbox.querySelector(".chatbox__messages");
    // Append HTML to the end of the chatbox
    chatmessage.innerHTML = html;
  }
}

const chatbox = new Chatbox();
chatbox.display();
