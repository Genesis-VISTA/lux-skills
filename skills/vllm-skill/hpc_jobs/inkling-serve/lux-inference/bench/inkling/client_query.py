import re
import sys
from openai import OpenAI

# USER NEEDS TO SET THESE
# all llama models
#temperature, top_p, systemprompt = 0.6, 0.9, "You are a knowledgeable, efficient, and direct AI assistant. Provide concise answers, focusing on the key information needed. Offer suggestions tactfully when appropriate to improve outcomes. Engage in productive collaboration with the user."

# ideal inference hyperparams for gptoss
temperature, top_p, systemprompt = 1., 1., "Reasoning: high"

# check if a hostname was provided as first argument
if len(sys.argv) > 1:
  servername = sys.argv[1]
# otherwise default to localhost
else:
  servername = "localhost"

# RZ persistent server
if(servername.startswith('rz')):
  http_prefix = "https://"
  portnum = ":443/v1"
else:
  http_prefix = "http://"
  portnum = ":8000/v1"

# turn on streaming for real-time inspection of results / debugging
# otherwise streaming=false for returing a single llm completion to a function
#streaming=False
streaming=True

# user-defined prompt
#userprompt = "What is the best way to calculate the first million digits of pi?"
#userprompt = "Write a python function to calculate the Wigner 3j symbols via recursion."
#userprompt = "Calculate the Ricci tensor for a two neutron star merger, assume that one neutron star has twice the mass of the other.  Next, calculate the stress-energy tensor and derive an equation for the geodesic motion."
#userprompt = "I want you to solve a problem in numerical general relativity by constructing the proper mathematical expressions, formulate a numerical approach to solve them, and then generate code that performs the actual numerical solution.  Consider a neutron star that is equal to the mass of earth's sun with a density distribution that is spherical.  Consider a second neutron star, with mass equal to that of earth's sun, also with a density distribution that is spherical, located initially at a distance of 100 times the neutron star radius.  The angular momentum of each neutron star is zero.  One neutron star is orbiting the other with a velocity of 1 percent the speed of light.   Assume that the matter of each neutron star can be approximated by a gamma law equation of state.  Generate a code for the Einstein Toolkit in C++ that will simulate the time evolution for this system of neutron stars merging to form a black hole. The code should be capable of numerically simulating this two neutron star merging system by solving the Einstein field equations, with calculation of the requisite stress-energy tensor, Riemann curvature tensor, metric tensor, Christoffel symbols and time integrate the geodesic equation formward with an explicit, discrete time-stepping scheme. Pay attention to stability of the dynamics, which is a notoriously difficult aspect of numerical GR, and handle the possibility of shock waves forming.  I want direct numerical simulation of the true 3D system using the ADM formalism."
#userprompt = "What is so special about the cellular automata rule 30?"
#userprompt = "Generate the first 1024 iterations of cellular automata rule 30 and display the sequence"
#userprompt = "Write a c++ code for the molecular dynamics simulation of a Lennard Jones system of 256 particles."
#userprompt = "what is the capital of france?"
#userprompt = "when solving problems in many body physics via Feynman diagrams, explain to me what is the significance of summing over clamshell-type diagrams?"
#userprompt = "Solving for the ground state energy of a hydrogen atom is rather straight forward in the Schrodinger or Heisenberg formulation of quantum mechanics, and is one of the very first problems that was solved when QM was a new theory.  The path integral formulation of quantum mechanics is physically equivalent to the Schrodinger equation, but expresses the theory in different mathematical terms.  The hydrogen atom solution via path integrals is somewhat more complicated, and it was several decades before it was solved by Duru and Kleinert.  Given that path integrals and the SE are isomorphic to the same theory, why is solving the hydrogen atom so difficult in the path integral formulation?"
#userprompt = "Renormalization group theory has been highly successful for calculating the critical temperature of phase transitions that are second order.  Can renormalization group theory also be applied to find the freezing temperature of a first order phase transition?"
#userprompt = "Can the KPZ equation be derived from Ginzberg-Landau theory?  If so, derive it for me."
#userprompt = "How can I construct a lattice boltzmann simulation for a fluid with an arbitrary equation of state?"
#userprompt = "On $\\triangle ABC$ points $A,D,E$, and $B$ lie that order on side $\\overline{AB}$ with $AD=4, DE=16$, and $EB=8$. Points $A,F,G$, and $C$ lie in that order on side $\\overline{AC}$ with $AF=13, FG=52$, and $GC=26$. Let $M$ be the reflection of $D$ through $F$, and let $N$ be the reflection of $G$ through $E$. Quadrilateral $DEGF$ has area 288. Find the area of heptagon $AFNBCEM$. Please reason step by step, and put your final answer within \\boxed{}"
#userprompt = "Explain to me the Feynman checkboard and how that is derived from the Dirac equation for electrodynamics in one dimension of space and one dimension of time.  Then, derive a cellular automata rule equivalent to the Feynman checkboard but for two dimensions of space and one dimension of time and then generate a numerical implementation in python."
#userprompt = "An unstoppable trolley is coming down the railroad tracks to a junction point at which one can use a manual switch to divert the trolley from one track to the other. Several adults are tied to the main track. An infant is tied to the alternative track. If you flip the switch, the infant will be killed. If you do not, several adults will be killed. No other options are possible. The switch determines whose life or lives you save and who you condemn? You control the switch. Do you push the switch or not?"
#userprompt = "What happens to the shock speed if I double the energy in a Sedov-Taylor blast?"
userprompt = "Explain why a high-Z dopant in the ICF ablator is used to suppress M-band preheat."
#userprompt = "How can I use machine learning to suppress Richtmyer-Meshkov instability?"
#userprompt = "Tell me about Jon Belof's work in suppressing Richtmyer-Meshkov instability"
#userprompt = "What are the pros and cons of the fast ignition concept of ICF"
#userprompt = "Describe when ICF ignition was first achieved on earth and how it was done"
#userprompt = "Describe how Dane Sterbentz was able to suppress Richtmyer-Meshkov instability"
#userprompt = "What was the purpose of the high foot capsule design and how did it contribute to ignition?"

# concatenate to form the total prompt
prompt = systemprompt + "\n" + userprompt

print("prompt input:\n")
print(f"{prompt}")
print("")

# point to the vllm openai API server
openai_api_key = "EMPTY"
openai_api_base = http_prefix + servername + portnum
print(f"connecting with API base of {openai_api_base}")
client = OpenAI(api_key=openai_api_key,base_url=openai_api_base,timeout=3600)

# get the list of models being served at the endpoint
models = client.models.list()
if models.data:
  #served_model = models.data[0]
  # take the last model in the list, useful if there are LoRA adapted models present
  served_model = models.data[-1]
  model = served_model.id
  print(f"found model {model} on {openai_api_base}")
else:
  print(f"error: no models found on {openai_api_base}")

# pass in our prompt
# set a seed and lower temperature for reproducibility
# total context buffer length is number of input tokens + max_tokens
response = client.completions.create(prompt=prompt, model=model, temperature=temperature, top_p=top_p, max_tokens=10000, user="Mastodon user", stream=streaming)

# output token size info
if(streaming == False):
    print("input prompt is ", response.usage.prompt_tokens , "tokens")
    print("output response is ", response.usage.completion_tokens, "tokens out of the total token output buffer of ", response.usage.total_tokens)

# print the response from the llm
print("llm response:\n")
if(streaming == True):
    for chunk in response:
      print(chunk.choices[0].text or "", end="")
else:
  print(response.choices[0].text)
print("")


