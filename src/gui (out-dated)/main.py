import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QHBoxLayout, QPushButton, QLabel, QFileDialog, QMessageBox
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtCore import QUrl, pyqtSlot
from klasseDefinities import Link, Linkketen, Telling
from functies import findStartLinks, createLinkKetens
import json
import random

class LinkVisualizerGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.links = []
        self.linkketens = []
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Link en Linkketen Visualizer')
        self.setGeometry(100, 100, 1200, 800)

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        main_layout = QVBoxLayout(central_widget)

        # Control panel
        control_panel = QHBoxLayout()

        self.load_data_btn = QPushButton('Laad Test Data')
        self.load_data_btn.clicked.connect(self.load_test_data)
        control_panel.addWidget(self.load_data_btn)

        self.create_linkketens_btn = QPushButton('Maak Linkketens')
        self.create_linkketens_btn.clicked.connect(self.create_linkketens)
        self.create_linkketens_btn.setEnabled(False)
        control_panel.addWidget(self.create_linkketens_btn)

        self.visualize_btn = QPushButton('Visualiseer op Kaart')
        self.visualize_btn.clicked.connect(self.visualize_data)
        self.visualize_btn.setEnabled(False)
        control_panel.addWidget(self.visualize_btn)

        control_panel.addStretch()

        self.status_label = QLabel('Gereed om data te laden')
        control_panel.addWidget(self.status_label)

        main_layout.addLayout(control_panel)

        # Web view for map
        self.web_view = QWebEngineView()
        main_layout.addWidget(self.web_view)

        # Load initial empty map
        self.load_empty_map()

    def load_empty_map(self):
        """Load an empty Leaflet map"""
        html_content = self.generate_map_html([], [])
        self.web_view.setHtml(html_content)

    def load_test_data(self):
        """Load some test data for demonstration"""
        try:
            # Create some test links
            self.links = [
                Link(1, 2, 'L001'),
                Link(2, 3, 'L002'),
                Link(3, 4, 'L003'),
                Link(4, 5, 'L004'),
                Link(2, 6, 'L005'),
                Link(6, 7, 'L006'),
                Link(5, 8, 'L007'),
                Link(7, 8, 'L008')
            ]

            # Set up link relationships
            for link in self.links:
                link.findPrevLinks(self.links)
                link.findNextLinks(self.links)

            # Add some dummy intensity data
            for i, link in enumerate(self.links):
                # Simulate some tellingen
                telling = Telling(f'LOC{i+1}', random.randint(100, 500), random.randint(20, 100))
                link.tellingenList = [telling]
                link.berekenIntensiteiten()

            self.status_label.setText(f'{len(self.links)} links geladen')
            self.create_linkketens_btn.setEnabled(True)

        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Fout bij laden van test data: {str(e)}')

    def create_linkketens(self):
        """Create linkketens from the loaded links"""
        try:
            start_links = findStartLinks(self.links)
            self.linkketens = createLinkKetens(start_links)

            # Set up linkketen relationships
            for linkketen in self.linkketens:
                linkketen.findPrevLinkketens(self.linkketens)
                linkketen.findNextLinkketens(self.linkketens)
                linkketen.bepaalBemeting()
                linkketen.berekenInitieleTelling()

            self.status_label.setText(f'{len(self.linkketens)} linkketens gemaakt')
            self.visualize_btn.setEnabled(True)

        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Fout bij maken van linkketens: {str(e)}')

    def visualize_data(self):
        """Visualize links and linkketens on the map"""
        try:
            html_content = self.generate_map_html(self.links, self.linkketens)
            self.web_view.setHtml(html_content)
            self.status_label.setText('Data gevisualiseerd op kaart')

        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Fout bij visualiseren: {str(e)}')

    def generate_map_html(self, links, linkketens):
        """Generate HTML content for the Leaflet map"""

        # Generate node positions (for demo purposes, using a simple grid)
        node_positions = {}
        nodes = set()
        for link in links:
            nodes.add(link.startNode)
            nodes.add(link.endNode)

        # Create a simple grid layout for nodes
        nodes_list = sorted(list(nodes))
        for i, node in enumerate(nodes_list):
            # Simple grid positioning (you might want to use real coordinates)
            row = i // 3
            col = i % 3
            lat = 52.0 + row * 0.01  # Netherlands approximate center
            lon = 5.0 + col * 0.01
            node_positions[node] = [lat, lon]

        # Generate JavaScript data for links
        links_js_data = []
        for link in links:
            if link.startNode in node_positions and link.endNode in node_positions:
                link_data = {
                    'id': link.linkNo,
                    'start': node_positions[link.startNode],
                    'end': node_positions[link.endNode],
                    'startNode': link.startNode,
                    'endNode': link.endNode,
                    'intensiteit': {
                        'PW': link.intensiteiten.get('PW', 0),
                        'VR': link.intensiteiten.get('VR', 0),
                        'Totaal': link.intensiteiten.get('Totaal', 0)
                    }
                }
                links_js_data.append(link_data)

        # Generate JavaScript data for linkketens
        linkketens_js_data = []
        for i, linkketen in enumerate(linkketens):
            linkketen_links = []
            for link in linkketen.links:
                if link.startNode in node_positions and link.endNode in node_positions:
                    linkketen_links.append({
                        'start': node_positions[link.startNode],
                        'end': node_positions[link.endNode],
                        'id': link.linkNo
                    })

            if linkketen_links:
                linkketen_data = {
                    'id': f'LK{i+1}',
                    'links': linkketen_links,
                    'bemeten': linkketen.bemeten,
                    'intensiteit': linkketen.intensiteiten.get('0-init', {'PW': 0, 'VR': 0, 'Totaal': 0})
                }
                linkketens_js_data.append(linkketen_data)

        html_template = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Link en Linkketen Visualizer</title>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.7.1/dist/leaflet.css" />
    <style>
        #map {{ height: 100vh; }}
        .link-popup {{
            font-family: Arial, sans-serif;
            font-size: 12px;
        }}
        .linkketen-popup {{
            font-family: Arial, sans-serif;
            font-size: 12px;
            background-color: #f0f8ff;
        }}
    </style>
</head>
<body>
    <div id="map"></div>

    <script src="https://unpkg.com/leaflet@1.7.1/dist/leaflet.js"></script>
    <script>
        // Initialize map
        var map = L.map('map').setView([52.0, 5.0], 10);

        // Add tile layer
        L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
            attribution: '© OpenStreetMap contributors'
        }}).addTo(map);

        // Links data
        var linksData = {json.dumps(links_js_data)};

        // Linkketens data
        var linkketensData = {json.dumps(linkketens_js_data)};

        // Color palette for linkketens
        var colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F'];

        // Draw individual links
        linksData.forEach(function(link, index) {{
            var polyline = L.polyline([link.start, link.end], {{
                color: '#666666',
                weight: 2,
                opacity: 0.7
            }}).addTo(map);

            var popupContent = `
                <div class="link-popup">
                    <h4>Link ${{link.id}}</h4>
                    <p><strong>Van:</strong> Node ${{link.startNode}}</p>
                    <p><strong>Naar:</strong> Node ${{link.endNode}}</p>
                    <p><strong>Intensiteit PW:</strong> ${{link.intensiteit.PW ? link.intensiteit.PW.toFixed(0) : 'N/A'}}</p>
                    <p><strong>Intensiteit VR:</strong> ${{link.intensiteit.VR ? link.intensiteit.VR.toFixed(0) : 'N/A'}}</p>
                    <p><strong>Totaal:</strong> ${{link.intensiteit.Totaal ? link.intensiteit.Totaal.toFixed(0) : 'N/A'}}</p>
                </div>
            `;

            polyline.bindPopup(popupContent);
        }});

        // Draw linkketens
        linkketensData.forEach(function(linkketen, index) {{
            var color = colors[index % colors.length];

            linkketen.links.forEach(function(link) {{
                var polyline = L.polyline([link.start, link.end], {{
                    color: color,
                    weight: 4,
                    opacity: 0.8
                }}).addTo(map);

                var popupContent = `
                    <div class="linkketen-popup">
                        <h4>Linkketen ${{linkketen.id}}</h4>
                        <p><strong>Link:</strong> ${{link.id}}</p>
                        <p><strong>Bemeten:</strong> ${{linkketen.bemeten ? 'Ja' : 'Nee'}}</p>
                        <p><strong>Intensiteit PW:</strong> ${{linkketen.intensiteit.PW ? linkketen.intensiteit.PW.toFixed(0) : 'N/A'}}</p>
                        <p><strong>Intensiteit VR:</strong> ${{linkketen.intensiteit.VR ? linkketen.intensiteit.VR.toFixed(0) : 'N/A'}}</p>
                        <p><strong>Totaal:</strong> ${{linkketen.intensiteit.Totaal ? linkketen.intensiteit.Totaal.toFixed(0) : 'N/A'}}</p>
                    </div>
                `;

                polyline.bindPopup(popupContent);
            }});
        }});

        // Add legend
        var legend = L.control({{position: 'bottomright'}});
        legend.onAdd = function(map) {{
            var div = L.DomUtil.create('div', 'info legend');
            div.innerHTML = `
                <div style="background-color: white; padding: 10px; border-radius: 5px; box-shadow: 0 0 15px rgba(0,0,0,0.2);">
                    <h4>Legenda</h4>
                    <p><span style="color: #666666; font-weight: bold;">━━━</span> Individuele Links</p>
                    <p><span style="color: #FF6B6B; font-weight: bold;">━━━</span> Linkketens (verschillende kleuren)</p>
                </div>
            `;
            return div;
        }};
        legend.addTo(map);

        // Fit map to show all data
        if (linksData.length > 0) {{
            var group = new L.featureGroup();
            linksData.forEach(function(link) {{
                group.addLayer(L.polyline([link.start, link.end]));
            }});
            map.fitBounds(group.getBounds().pad(0.1));
        }}
    </script>
</body>
</html>
        """

        return html_template

def main():
    app = QApplication(sys.argv)
    window = LinkVisualizerGUI()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()