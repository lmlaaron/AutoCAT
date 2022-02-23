import pandas as pd 

# def read_file
# def parser_action()

number_of_set = 2

def get_set(addr):

	return addr%number_of_set

def get_order(addr): # return (addr)/number_of_set

  input = pd.DataFrame(test)
  df = input.astype('int')

   # each row indicates the list of [(attacker) address, is_guess, is_victim, is_flush, victim_addr]
  df.columns =['addr', 'is_guess', 'is_victim', 'is_flush', 'victim_addr']
  df['set'] = df(get_set)

  df_set0 = df[df.set==0]
  df_set0['order'] = df_set0['addr'].rank(method='dense',ascending=False).astype(int)
  print(df_set0)

  df_set1 = df[df.set==1]
  df_set1['order'] = df_set1['addr'].rank(method='dense',ascending=True).astype(int)
  print(df_set1)

  frames = [df_set0, df_set1]
  result = pd.concat(frames)
  df = pd.DataFrame(result)
  df =df.sort_index(axis=0, ascending=True)

def rename_addr(df): # rename the addres in the pattern based on the set and the order appeared in the pattern 
# output = [#set, #the order the address appear in the attack, is_guess, is_victim, is_flush, victim_addr]
  df = df[['set','order','is_guess', 'is_victim', 'is_flush', 'victim_addr']] 
  return df

def remove(df): # remove repeated access

  return df.drop_duplicates()

# Defining main function
def main():
  test = [[1, 0, 0, 0, 0], [3, 0, 0, 0, 0], [4, 0, 0, 0, 0], [1, 0, 0, 0, 0], [5, 0, 0, 0, 0], [0, 0, 1, 0, 0], [1, 0, 0, 0, 0], [3, 0, 0, 0, 0]]
  
  input = pd.DataFrame(test)  
  df = input.astype('int')
   
  df = rename_addr(df) 
  df = remove(df)
  print(df)

# Using the special variable 
# __name__
if __name__=="__main__":
    main()