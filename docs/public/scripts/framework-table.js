(function () {
  var LEGEND = {
    "Callable Tools":
      "Supports using Python callables (functions, classes) directly as agent tools.",
    "MCP Tools":
      "Supports <a href='https://modelcontextprotocol.io/' target='_blank'>Model Context Protocol</a> tools (Stdio, SSE, Streamable HTTP).",
    "Composio Tools":
      "Supports <a href='https://composio.dev/' target='_blank'>Composio</a> tool integration.",
    "Structured Output":
      "Supports constraining agent output to a specific schema (e.g., Pydantic model).",
    Streaming: "Supports streaming agent responses as they are generated.",
    "Multi-Agent (Handoffs)":
      "Supports native multi-agent handoff patterns within the framework.",
    Callbacks:
      "Supports the any-agent <a href='/any-agent/agents/callbacks/'>callback system</a> for lifecycle hooks.",
    "any-llm Integration":
      "Uses <a href='https://docs.mozilla.ai/any-llm/' target='_blank'>any-llm</a> as the unified model provider.",
  };

  document.addEventListener("DOMContentLoaded", function () {
    var input = document.getElementById("framework-search");
    if (!input) return;

    var content = document.querySelector(".sl-markdown-content");
    if (!content) return;

    var table = content.querySelector("table");
    if (!table) return;

    // Parse headers
    var headerCells = Array.from(table.querySelectorAll("thead th"));
    if (headerCells.length === 0) {
      headerCells = Array.from(
        table.querySelector("tr").querySelectorAll("th")
      );
    }
    var headers = headerCells.map(function (th) {
      return th.textContent.trim();
    });

    // Parse rows into framework objects
    var frameworks = [];
    var dataRows = Array.from(table.querySelectorAll("tbody tr"));
    if (dataRows.length === 0) {
      dataRows = Array.from(table.querySelectorAll("tr")).filter(function (
        row
      ) {
        return row.querySelector("td");
      });
    }

    dataRows.forEach(function (row) {
      var cells = Array.from(row.querySelectorAll("td"));
      var nameCell = cells[0];
      var link = nameCell ? nameCell.querySelector("a") : null;

      var framework = {
        name: nameCell ? nameCell.textContent.trim() : "",
        url: link ? link.href : null,
        config: [],
        features: [],
        searchText: row.textContent.toLowerCase(),
      };

      // Config columns (Docs link)
      for (var i = 1; i <= 1 && i < cells.length; i++) {
        var configLink = cells[i] ? cells[i].querySelector("a") : null;
        framework.config.push({
          label: headers[i],
          value: configLink ? configLink.href : cells[i].textContent.trim(),
        });
      }

      // Feature columns
      for (var j = 2; j < headers.length && j < cells.length; j++) {
        framework.features.push({
          label: headers[j],
          supported: cells[j].textContent.indexOf("\u2705") !== -1,
        });
      }

      frameworks.push(framework);
    });

    // Hide the original table
    var tableEl =
      table.parentElement !== content &&
      table.parentElement.tagName === "DIV"
        ? table.parentElement
        : table;
    tableEl.style.display = "none";

    // Update framework count in intro text
    var intro = content.querySelector("p");
    if (intro && intro.innerHTML.indexOf("{n}") !== -1) {
      intro.innerHTML = intro.innerHTML.replace("{n}", frameworks.length);
    }

    // Build framework grid
    var grid = document.createElement("div");
    grid.className = "framework-grid";

    var cards = [];
    frameworks.forEach(function (framework) {
      var card = document.createElement("button");
      card.className = "framework-card";
      card.type = "button";

      var supported = framework.features.filter(function (f) {
        return f.supported;
      }).length;
      var total = framework.features.length;

      // Dots row showing feature support at a glance
      var dots = framework.features
        .map(function (f) {
          return (
            '<span class="framework-card-dot ' +
            (f.supported ? "dot-yes" : "dot-no") +
            '"></span>'
          );
        })
        .join("");

      card.innerHTML =
        '<span class="framework-card-name">' +
        framework.name +
        "</span>" +
        '<span class="framework-card-dots">' +
        dots +
        "</span>" +
        '<span class="framework-card-count">' +
        supported +
        "/" +
        total +
        " features</span>";

      card.addEventListener("click", function () {
        showModal(framework);
      });

      grid.appendChild(card);
      cards.push({ el: card, framework: framework });
    });

    var searchContainer = input.closest(".framework-search-container");
    searchContainer.parentNode.insertBefore(grid, searchContainer.nextSibling);

    // No results
    var noResults = document.createElement("p");
    noResults.textContent = "No frameworks match your search.";
    noResults.className = "framework-no-results";
    noResults.style.display = "none";
    grid.parentNode.insertBefore(noResults, grid.nextSibling);

    // Modal
    var overlay = document.createElement("div");
    overlay.className = "framework-modal-overlay";
    overlay.style.display = "none";
    overlay.addEventListener("click", function (e) {
      if (e.target === overlay) closeModal();
    });
    var modalEl = document.createElement("div");
    modalEl.className = "framework-modal";
    overlay.appendChild(modalEl);
    document.body.appendChild(overlay);

    document.addEventListener("keydown", function (e) {
      if (e.key === "Escape") closeModal();
    });

    function closeModal() {
      overlay.style.display = "none";
      document.body.style.overflow = "";
    }

    function showModal(framework) {
      var html =
        '<button class="framework-modal-close" aria-label="Close">&times;</button>';

      html += '<h2 class="framework-modal-title">';
      html += framework.url
        ? '<a href="' +
          framework.url +
          '">' +
          framework.name +
          " &#8599;</a>"
        : framework.name;
      html += "</h2>";

      // Features
      html += '<div class="framework-modal-features">';
      framework.features.forEach(function (f) {
        html +=
          '<div class="framework-modal-feature ' +
          (f.supported ? "supported" : "unsupported") +
          '">' +
          '<span class="framework-modal-dot ' +
          (f.supported ? "dot-yes" : "dot-no") +
          '"></span>' +
          "<span>" +
          f.label +
          "</span></div>";
      });
      html += "</div>";

      // Configuration
      html +=
        '<h3>Configuration</h3><dl class="framework-modal-config">';
      framework.config.forEach(function (c) {
        html += "<dt>" + c.label + "</dt>";
        if (c.value && c.value.startsWith("http")) {
          html +=
            "<dd><a href='" +
            c.value +
            "' target='_blank'>" +
            c.value +
            "</a></dd>";
        } else {
          html += "<dd>" + (c.value || "\u2014") + "</dd>";
        }
      });
      html += "</dl>";

      // Legend
      html +=
        '<details class="framework-modal-legend"><summary>What do these features mean?</summary><dl>';
      framework.features.forEach(function (f) {
        if (LEGEND[f.label]) {
          html +=
            "<dt>" + f.label + "</dt><dd>" + LEGEND[f.label] + "</dd>";
        }
      });
      html += "</dl></details>";

      modalEl.innerHTML = html;
      overlay.style.display = "flex";
      document.body.style.overflow = "hidden";

      modalEl
        .querySelector(".framework-modal-close")
        .addEventListener("click", closeModal);
    }

    // Search
    input.addEventListener("input", function () {
      var query = input.value.toLowerCase().trim();
      var visibleCount = 0;

      cards.forEach(function (item) {
        if (!query) {
          item.el.style.display = "";
          visibleCount++;
          return;
        }
        var match = item.framework.searchText.includes(query);
        item.el.style.display = match ? "" : "none";
        if (match) visibleCount++;
      });

      noResults.style.display = visibleCount === 0 ? "block" : "none";
    });
  });
})();
