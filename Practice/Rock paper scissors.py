from random import randint
def getpl_choices():
    player_choice= input("Enter your hand")
    try:
        print("This is a Rock, Paper & Scissors game against computer")
        if player_choice.lower() == "rock" or player_choice.lower() == "scissors" or player_choice.lower() == "paper":
            pass
        else:
            print("worng input ")
            raise ValueError
    except ValueError:
        print("You have entered a wrong hand try again!!!")
        getpl_choices()
    return player_choice.lower()


def comp_hand():
    hand_list = {
        1:"rock",
        2:"paper",
        3:"scissors"
    }
    gethand= randint(1,3);
    hand = hand_list[gethand]
    return hand.lower()


def compare_hand(player_choice, computer_choice):
    if player_choice == computer_choice:
        print ("Draw ")
        pass
    elif player_choice =="rock":
        if computer_choice == "paper":
            print(" Computer wins!")
            pass
        elif computer_choice == "scissors":
            print ("Player wins!")
            pass
        pass
    elif player_choice =="scissors":
        if computer_choice == "rock":
            print("Computer wins!")
            pass
        elif computer_choice == "paper":
            print("Player wins!")
            pass
        pass
    elif player_choice == "paper":
        if computer_choice == "rock":
            print("Player wins")
            pass
        elif computer_choice == "scissors":
            print ("Computer wins!")
            pass
        pass

choice= input("do you want to start ? yes or no   ")
while choice.lower() == "yes":
    hand =getpl_choices()
    computerhand= comp_hand()
    print("The computers hand is :- " + computerhand)
    compare_hand(hand, computerhand)
    choice = input("Do you want to continue ? yes or no:  ")
if choice.lower() == "no":
    print ("Thanks for playing!!!")