import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.widgets import Button
import noisy_graph_states as nsf
import noisy_graph_states.libs.graph as gt
import noisy_graph_states.libs.matrix as mat
import pyperclip  # Importa pyperclip per gestire gli appunti

mesure=[]
p = "1"
fid=""
# Funzione di local complementazione (semplificata)
def local_complementation(graph,n):
    neighbors = list(graph.neighbors(n))
    for i in range(len(neighbors)):
        for j in range(i + 1, len(neighbors)):
            if graph.has_edge(neighbors[i], neighbors[j]):
                graph.remove_edge(neighbors[i], neighbors[j])
            else:
                graph.add_edge(neighbors[i], neighbors[j])
   # Registra l'operazione nella lista 'mesure'    
    return graph
# Funzione di local Pauli Z che rimuove tutti i collegamenti del nodo
def local_PauliZ(graph, n):
    print(f"Rimuovendo tutti i collegamenti del nodo {label_map[n]}")
    graph.remove_edges_from(list(graph.edges(n)))
    node_colors[n] = 'red'  # Colore rosso per le misure Z
     # Registra l'operazione nella lista 'mesure'
    return graph

# Funzione di local Pauli Y
def local_PauliY(graph, n):
    print(f"Eseguendo la misurazione locale di Pauli Y sul nodo {label_map[n]}")
    loc_graph = local_complementation(graph,n)
    loc_graph = local_PauliZ(loc_graph, n)
    node_colors[n] = 'darkblue'  # Colore blu scuro per le misure Y

    return loc_graph

# Funzione di local Pauli X
def local_PauliX(graph, n, neighbor=-1):
    neighbors = list(graph.neighbors(n))
    print(f"Vicini di {label_map[n]}: {[label_map[nb] for nb in neighbors]}")

    if neighbor == -1:
        neighbor = neighbors[0]
    if neighbor not in neighbors:
        raise ValueError(f"Il nodo {label_map[neighbor]} non è un vicino del nodo {label_map[n]}.")

    loc_graph = local_complementation(graph,neighbor )
    loc_graph = local_PauliY(loc_graph, n)
    loc_graph = local_complementation(loc_graph,neighbor )
    node_colors[n] = 'yellow'
    node_colors[neighbor] = 'green'

    return loc_graph

# Funzione per la complementazione locale su un percorso
def local_compl_on_Path(graph, sequence):
    loc_com = graph
    mesure.append(("lc", label_map[sequence[0]]))
    for i in sequence:
        loc_com = local_complementation(loc_com,i)
        

    for i in sequence[1:-1]:
        loc_com = local_PauliZ(loc_com, i)
        node_colors[i] = 'yellow'
        mesure.append(("y", label_map[i]))
    node_colors[sequence[0]] = 'green'
    mesure.append(("lc", label_map[sequence[0]]))
    return loc_com

# Funzione per l'operazione di undo
def undo(event):
    global G, node_colors, stack, selected_nodes,mesure
    if stack:
        G, node_colors, selected_nodes, measurements = stack.pop()
        ax.clear()
        draw_grid_graph(G, N)
        plt.draw()
        print("Operazione di Undo eseguita.")

# Funzione per stampare e visualizzare i vicini di un nodo
def neighbors(graph, n):
    neighbor_labels = [label_map[neighbor] for neighbor in graph.neighbors(n)]
    print(f"I vicini del nodo {label_map[n]} sono: {neighbor_labels}")
    return label_map[n],neighbor_labels
# Funzione per stampare e visualizzare per visualizare tutte le misure effetuate
def estraction_mesure(event):
    print("le misure effetuate sono:",mesure)
    mesurecopy.set_text(f'{mesure}')  # new text
    plt.draw()



# Funzione per trovare tutti i cammini tra due nodi
def find_good_paths(graph, source, target):
    # Trova tutti i cammini semplici tra source e target
    lenght=abs(source[0]-target[0])+abs(source[0]-target[0])+4
    all_paths = list(nx.all_simple_paths(graph, source=source, target=target,cutoff=lenght))
    
    # Filtra i cammini con lunghezza inferiore a 4
    for i, path in enumerate(all_paths):
        print(f"Cammino {i + 1}: {[label_map[node] for node in path]}")
    
    return all_paths
def run(graph, nodes,p):
    
    pr=float(p)
    Tgraph=graph.copy()
    Tgraph=nx.convert_node_labels_to_integers(Tgraph)
    
    
    input_state = nsf.State(Tgraph, [])
    input_state = nsf.pauli_noise(input_state, range(len(Tgraph)), [pr + (1 - pr) / 4, (1 - pr) / 4, (1 - pr) / 4, (1 - pr) / 4])
    

    strat = nsf.Strategy(Tgraph, mesure)
    output = strat(input_state)
    strat.save()
    fid = np.real_if_close(
                fidelity(
                    gt.bell_pair_ket, nsf.noisy_bp_dm(output, target_indices=nodes)
                )
             )[0, 0]

    return fid 

import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import pyperclip  # Importa pyperclip per gestire gli appunti

# Funzione di callback per il pulsante
def copy_to_clipboard(event):
    text_to_copy = str(mesure)  # Testo da copiare
    pyperclip.copy(text_to_copy)  # Copia il testo negli appunti
    print(f"Testo '{text_to_copy}' copiato negli appunti!")



N = 10
G = nx.grid_2d_graph(N, N)
G0=G.copy()
node_colors = {node: 'lightblue' for node in G.nodes()}
def update_and_clear(selected_nodes):
    # Cambia il colore dei nodi selezionati a 'lightblue' e svuota la lista dei nodi selezionati in una riga
    [node_colors.update({node: 'lightblue'}) for node in selected_nodes] 
    selected_nodes.clear()
   
    draw_grid_graph(G, N)
    plt.draw()

# neighbors_active = False  # Variabile globale per tracciare lo stato della funzione neighbors
# path_mode_active = False  # Variabile globale per tracciare lo stato della modalità di percorso
selected_nodes = []  # Lista per tenere traccia dei nodi selezionati per il percorso

# Mappa per associare i nodi ai loro label
label_map = {}
count = 0
for j in range(N-1, -1, -1):
    for i in range(N):
        label_map[(i, j)] = count
        count += 1

stack = []
current_function = local_PauliY
selected_nodes = []
nodo=''
vicinonodo=''

def on_click(event):
    global G0, G, node_colors, selected_nodes, pos, selected_nodes,mesure
    
    if event.inaxes == ax:
        x = int(round(event.xdata))
        y = int(round(event.ydata))
        
        # Trovare il nodo più vicino al punto cliccato (questo sarà una tupla (x, y))
        node = min(G.nodes, key=lambda n: np.linalg.norm(np.array(pos[n]) - np.array([x, y])))
        # Salva lo stato attuale per la funzione undo
        stack.append((G.copy(), node_colors.copy(), selected_nodes.copy(),mesure.copy()))

        if node is not None:
            print(f"Hai selezionato il nodo {label_map[node]}")

            if current_function == local_compl_on_Path:
                if selected_nodes and node == selected_nodes[0]:
                    selected_nodes.append(node)
                    G = local_compl_on_Path(G, selected_nodes)
                    print("your path is:", selected_nodes)
                    selected_nodes.clear()
                    ax.clear()
                    draw_grid_graph(G, N)
                    plt.draw()
                else:
                    selected_nodes.append(node)
                    node_colors[node] = 'yellow'
                    node_colors[selected_nodes[0]] = 'green'
                    draw_grid_graph(G, N)                    
                    plt.draw()
            elif current_function == local_PauliX:
                selected_nodes.append(node)
                if len(selected_nodes) == 1:
                    node_colors[node] = 'yellow'
                draw_grid_graph(G, N)
                plt.draw()

                if len(selected_nodes) == 2:
                    local_PauliX(G, selected_nodes[0], selected_nodes[1])
                    mesure.append(("x", label_map[selected_nodes[0]],label_map[selected_nodes[1]]))

                    selected_nodes.clear()
                    ax.clear()
                    draw_grid_graph(G, N)
                    plt.draw()
            elif current_function == find_good_paths:
                selected_nodes.append(node)
                node_colors[node] = 'purple'
                draw_grid_graph(G, N)
                plt.draw()
                if len(selected_nodes) == 2:
                    find_good_paths(G, selected_nodes[0], selected_nodes[1])
                    selected_nodes.clear()
                
            elif current_function == local_PauliY:
                G=local_PauliY(G, node)
                mesure.append(("y", label_map[node]))
                 # Ridisegna il grafo
                ax.clear()
                draw_grid_graph(G, N)
                plt.draw()
            
            elif current_function == local_PauliZ:
                G=local_PauliZ(G, node)
                mesure.append(("z", label_map[node]))
                 # Ridisegna il grafo
                ax.clear()
                draw_grid_graph(G, N)
                plt.draw()
            
            elif current_function == local_complementation:
                G=local_complementation(G, node)
                mesure.append(("lc", label_map[node]))
                 # Ridisegna il grafo
                ax.clear()
                draw_grid_graph(G, N)
                plt.draw()
            
            elif current_function == neighbors:
                nodo,vicinonodo=neighbors(G, node)
                list_neighbors.set_text(f'i vicini del nodo {nodo} sono:{vicinonodo}')  # Aggiorna il testo visualizzato
                plt.draw()
            elif current_function == run:
                selected_nodes.append(node)
                if len(selected_nodes) == 1:
                    node_colors[node] = 'green'
                draw_grid_graph(G, N)
                plt.draw()

                if len(selected_nodes) == 2:
                    selected_nodes_label=[label_map[selected_nodes[0]],label_map[selected_nodes[1]]]
                    fid=str(f"{run(G0,selected_nodes_label,p):.4f}")
                    fidel.set_text(f'fidelity={fid}')  # Aggiorna il testo visualizzato

                    print("your path is:", fid)
                    node_colors[selected_nodes[1]] = 'green'  
                    selected_nodes.clear()
                    ax.clear()
                    draw_grid_graph(G, N)
                    plt.draw()
    elif event.inaxes == ax_mesurecopy:
        copy_to_clipboard(event)

       
    elif event.inaxes == ax_pauli_y:
        select_pauli_y(event)
    elif event.inaxes == ax_pauli_z:
        select_pauli_z(event)
    elif event.inaxes == ax_pauli_x:
        select_pauli_x(event)
    elif event.inaxes == ax_path:
        select_path(event)
    elif event.inaxes == ax_local_complementation:
        select_local_complementation(event)
    elif event.inaxes == ax_find_paths:
        select_find_good_paths(event)
    elif event.inaxes == ax_neighbors:
        select_neighbors(event)
    elif event.inaxes == ax_run:
        select_run(event)


def fidelity(target_ket, rho):
    return mat.H(target_ket) @ rho @ target_ket


def generate_sequence(path, open_ends, source_idx):
    return tuple(("x", i, source_idx) for i in path) + tuple(
        ("z", i) for i in open_ends
    )
    
   
def draw_grid_graph(G, N):
    global pos, node_colors
    pos = {(x, y): (x, y) for x, y in G.nodes()}
    colors = [node_colors[node] for node in G.nodes()]
    labels = {node: label_map[node] for node in G.nodes()}
    nx.draw(G, pos=pos, labels=labels, with_labels=True, node_color=colors, node_size=700, font_size=10, edge_color='gray', ax=ax)

def select_pauli_y(event):
    global current_function, selected_nodes
    current_function = local_PauliY
    update_and_clear(selected_nodes)
    print("Funzione selezionata: Pauli Y")
    highlight_selection(ax_pauli_y, [ax_run, ax_pauli_z, ax_pauli_x,ax_neighbors,ax_find_paths, ax_path, ax_local_complementation])

def select_pauli_z(event):
    global current_function, selected_nodes
    current_function = local_PauliZ
    update_and_clear(selected_nodes)
    print("Funzione selezionata: Pauli Z")
    highlight_selection(ax_pauli_z, [ax_run, ax_pauli_y, ax_pauli_x,ax_neighbors,ax_find_paths, ax_path, ax_local_complementation])

def select_pauli_x(event):
    global current_function, selected_nodes
    current_function = local_PauliX
    update_and_clear(selected_nodes)
    print("Funzione selezionata: Pauli X")
    highlight_selection(ax_pauli_x, [ax_run, ax_pauli_y, ax_pauli_z,ax_neighbors,ax_find_paths, ax_path, ax_local_complementation])

def select_path(event):
    global current_function, selected_nodes
    current_function = local_compl_on_Path
    update_and_clear(selected_nodes)
    print("Funzione selezionata: Complementazione su Percorso")
    highlight_selection(ax_path , [ax_run, ax_pauli_y, ax_pauli_z,ax_neighbors,ax_find_paths, ax_pauli_x,ax_local_complementation ])

def select_local_complementation(event):
    global current_function, selected_nodes
    current_function = local_complementation
    update_and_clear(selected_nodes)
    print("Funzione selezionata: local complementation")
    highlight_selection(ax_local_complementation, [ax_run, ax_pauli_y,ax_neighbors,ax_find_paths, ax_pauli_z, ax_pauli_x, ax_path])

def select_neighbors(event):
    global current_function, selected_nodes
    current_function = neighbors
    update_and_clear(selected_nodes)
    print("Funzione selezionata: local complementation")
    highlight_selection(ax_neighbors, [ax_run, ax_pauli_y, ax_pauli_z, ax_pauli_x, ax_path,ax_local_complementation,ax_find_paths])


def select_find_good_paths(event):
    global current_function, selected_nodes
    current_function = find_good_paths
    update_and_clear(selected_nodes)
    print("Funzione selezionata: local complementation")
    highlight_selection(ax_find_paths, [ax_run, ax_pauli_y, ax_pauli_z, ax_pauli_x, ax_path,ax_local_complementation,ax_neighbors])

def select_run(event):
    global current_function, selected_nodes
    current_function = run
    update_and_clear(selected_nodes)
    print("Funzione selezionata: local complementation")
    highlight_selection(ax_run, [ax_find_paths, ax_pauli_y, ax_pauli_z, ax_pauli_x, ax_path,ax_local_complementation,ax_neighbors])


def highlight_selection(selected_ax, other_axes):
    # Cambia il colore del selezionato
    selected_ax.add_patch(Rectangle((0, 0), 1, 1, facecolor='green'))
    
    # Cambia il colore degli altri
    for ax in other_axes:
        ax.add_patch(Rectangle((0, 0), 1, 1, facecolor='lightgray'))
    
    plt.draw()  # Usa draw per aggiornare la visualizzazione

# Funzione di callback per il pulsante


fig, ax = plt.subplots()
plt.subplots_adjust(right=0.7)
draw_grid_graph(G, N)



ax_undo = plt.axes([0.01, 0.9, 0.1, 0.075])
btn_undo = Button(ax_undo, 'Undo')
btn_undo.on_clicked(undo)

ax_mesure = plt.axes([0.01, 0.7, 0.1, 0.075])
btn_mesure = Button(ax_mesure, 'extract mesure')
btn_mesure.on_clicked(estraction_mesure)



ax_mesurecopy = plt.axes([0.55, 0.9, 0.3, 0.08])
ax_mesurecopy.set_axis_off()  # Nascondi gli assi
ax_mesurecopy.add_patch(Rectangle((0, 0), 1, 1, facecolor='lightgray'))
mesurecopy=ax_mesurecopy.text(0.5, 0.5, 'le misure sono:', horizontalalignment='center', verticalalignment='center', transform=ax_mesurecopy.transAxes)




ax_neighbors = plt.axes([0.01, 0.8, 0.1, 0.075])
ax_run = plt.axes([0.01, 0.5, 0.1, 0.075])
ax_find_paths = plt.axes([0.01, 0.6, 0.1, 0.075])
ax_pauli_y = plt.axes([0.75, 0.75, 0.2, 0.08])
ax_pauli_z = plt.axes([0.75, 0.65, 0.2, 0.08])
ax_pauli_x = plt.axes([0.75, 0.55, 0.2, 0.08])
ax_path = plt.axes([0.75, 0.45, 0.2, 0.08])
ax_local_complementation = plt.axes([0.75, 0.35, 0.2, 0.08])
ax_list_neighbors= plt.axes([0.3, 0.9, 0.2, 0.08])
ax_fidelity= plt.axes([0.15, 0.9, 0.1, 0.08])


ax_pauli_y.set_axis_off()  # Nascondi gli assi
ax_pauli_z.set_axis_off()  # Nascondi gli assi
ax_pauli_x.set_axis_off()  # Nascondi gli assi
ax_path.set_axis_off()  # Nascondi gli assi
ax_local_complementation.set_axis_off()  # Nascondi gli assi
ax_neighbors.set_axis_off()  # Nascondi gli assi
ax_find_paths.set_axis_off()  # Nascondi gli assi
ax_list_neighbors.set_axis_off()  # Nascondi gli assi
ax_run.set_axis_off()  # Nascondi gli assi
ax_fidelity.set_axis_off()  # Nascondi gli assi


ax_pauli_y.add_patch(Rectangle((0, 0), 1, 1, facecolor='lightgray'))
ax_pauli_z.add_patch(Rectangle((0, 0), 1, 1, facecolor='lightgray'))
ax_pauli_x.add_patch(Rectangle((0, 0), 1, 1, facecolor='lightgray'))
ax_path.add_patch(Rectangle((0, 0), 1, 1, facecolor='lightgray'))
ax_local_complementation.add_patch(Rectangle((0, 0), 1, 1, facecolor='lightgray'))
ax_neighbors.add_patch(Rectangle((0, 0), 1, 1, facecolor='lightgray'))
ax_find_paths.add_patch(Rectangle((0, 0), 1, 1, facecolor='lightgray'))
ax_list_neighbors.add_patch(Rectangle((0, 0), 1, 1, facecolor='lightgray'))
ax_run.add_patch(Rectangle((0, 0), 1, 1, facecolor='lightgray'))
ax_fidelity.add_patch(Rectangle((0, 0), 1, 1, facecolor='lightgray'))

ax_pauli_y.text(0.5, 0.5, 'Pauli Y', horizontalalignment='center', verticalalignment='center', transform=ax_pauli_y.transAxes)
ax_pauli_z.text(0.5, 0.5, 'Pauli Z', horizontalalignment='center', verticalalignment='center', transform=ax_pauli_z.transAxes)
ax_pauli_x.text(0.5, 0.5, 'Pauli X', horizontalalignment='center', verticalalignment='center', transform=ax_pauli_x.transAxes)
ax_path.text(0.5, 0.5, 'Path', horizontalalignment='center', verticalalignment='center', transform=ax_path.transAxes)
ax_local_complementation.text(0.5, 0.5, 'local complementation', horizontalalignment='center', verticalalignment='center', transform=ax_local_complementation.transAxes)
ax_find_paths.text(0.5, 0.5, 'find good paths', horizontalalignment='center', verticalalignment='center', transform=ax_find_paths.transAxes)
ax_neighbors.text(0.5, 0.5, 'neighbors', horizontalalignment='center', verticalalignment='center', transform=ax_neighbors.transAxes)
list_neighbors=ax_list_neighbors.text(0.5, 0.5, 'i vicini del nodo   sono:' , horizontalalignment='center', verticalalignment='center', transform=ax_list_neighbors.transAxes)
ax_run.text(0.5, 0.5, 'noise bell pair', horizontalalignment='center', verticalalignment='center', transform=ax_run.transAxes)



fidel=ax_fidelity.text(0.5, 0.5, 'fidelity =', horizontalalignment='center', verticalalignment='center', transform=ax_fidelity.transAxes)



highlight_selection(ax_pauli_y, [ax_run, ax_pauli_z, ax_pauli_x,ax_neighbors,ax_find_paths, ax_path, ax_local_complementation])
fig.canvas.mpl_connect('button_press_event', on_click)




# Aggiungiamo un rettangolo e un testo al grafico
ax_noise =plt.axes([0.01, 0.1, 0.1, 0.075])
ax_noise.add_patch(Rectangle((0, 0), 1, 1, facecolor='lightgray'))
noise=ax_noise.text(0.1, 0.1, f'p={p}', verticalalignment='bottom', horizontalalignment='left', color='black', fontsize=15)
ax_noise.set_axis_off()  # Nascondi gli assi

def on_key(event):
    global p
    old_p = p  # Conserva il valore precedente di p in caso di necessità di rollback
    if event.key.isdigit() or (event.key == '.' and '.' not in p):
        p += event.key  # Aggiunge il carattere premuto alla stringa p
    elif event.key == 'backspace':
        p = p[:-1]  # Rimuove l'ultimo carattere

    # Verifica che p non superi 1 dopo ogni modifica
    try:
        if float(p) >1:
            p = old_p  # Ripristina p al valore precedente se supera 1
    except ValueError:
        pass  # Gestisce eccezioni per valori non numerici o stringhe vuote

    noise.set_text(f'p={p}')  # Aggiorna il testo visualizzato
    plt.draw()

# Collegamento dell'evento di pressione del tasto con la funzione on_key
fig.canvas.mpl_connect('key_press_event', on_key)








plt.show()
