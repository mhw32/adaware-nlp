'''
Make sure Neo4j database is running locally with default @ http://localhost:7474/db/data/

Set the username and password by going to http://localhost:747
Username: neo4j (DEFAULT)
Password: adaware

py2neo documentation @ http://py2neo.org/2.0/intro.html
'''
import os
import uuid
from py2neo import authenticate
from py2neo import Graph, Node, Relationship

username = os.environ.get('NEO4J_USERNAME') or 'neo4j'
password = os.environ.get('NEO4J_PASSWORD') or 'adaware'
url = 'http://localhost:7474'

if username and password:
    authenticate(url.strip('http://'), username, password)

graph = Graph(url + '/db/data/')

graph.cypher.execute("MATCH (n) DETACH DELETE n")

alice = Node("Person", name="Alice")
bob = Node("Person", name="Bob")
alice_knows_bob = Relationship(alice, "KNOWS", bob)

graph.create(alice_knows_bob)

alice.properties["age"] = 33
bob.properties["age"] = 44
alice.push()
bob.push()

for record in graph.cypher.execute("MATCH (p:Person) RETURN p.name AS name"):
    print(record[0])