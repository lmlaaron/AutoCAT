

# def read_file
# def parser_action()

def rename_addr():
# rename the addres in the pattern based on the set and the order appeared in the pattern 
# each row indicates the list of [(attacker) address, is_guess, is_victim, is_flush, victim_addr]

# below 'test' needs to be replaced by parser_action
  test = [[1, 0, 0, 0, 0], [3, 0, 0, 0, 0], [4, 0, 0, 0, 0], [1, 0, 0, 0, 0], [5, 0, 0, 0, 0], [0, 0, 1, 0, 0], [1, 0, 0, 0, 0], [3, 0, 0, 0, 0]]
  import pandas as pd 
  input = pd.DataFrame(test)
  df = input.astype('int')
  df.columns =['atk_addr', 'is_guess', 'is_victim', 'is_flush', 'victim_addr'] 
  df['set'] = df.atk_addr%2 #assumed this is 2-way associative cache

  def set1(x):
    if x%2!=0:
      return (x+1)/2
    elif x%2==0:
      return (x+2)/2

  df['order'] = df['atk_addr'].apply(set1).astype(int) #the order the address appear in the attack

  # output = [#set, #the order the address appear in the attack, is_guess, is_victim, is_flush, victim_addr]
  df = df[['set','order','is_guess', 'is_victim', 'is_flush', 'victim_addr']] 

def remove(): # remove repeated access
  df2= df.drop_duplicates()
  print(df2)

# Defining main function
def main():
    rename_addr() 
    remove()
  


# Using the special variable 
# __name__
if __name__=="__main__":
    main()


