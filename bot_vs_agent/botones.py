import retro

game = 'SamuraiShodown-Genesis'
state = 'Level1.HaohmaruVsHaohmaru'

env = retro.make(game=game, state=state)

print("=============== BOTONES DEL ENTORNO ===============")
print(env.buttons)
print("Cantidad de botones:", len(env.buttons))
print("====================================================")

env.close()
